import os
import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
import unicodedata
import re
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import textwrap
import torch
import time


# Funções Auxiliares
def check_cuda():
    return "CUDA disponível." if torch.cuda.is_available() else "CUDA não disponível."

def sanitize_filename(filename):
    """Remove caracteres especiais do nome do arquivo."""
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    return filename.replace(' ', '_')

def extract_text_from_pdf(reader):
    """Extrai texto de todas as páginas do PDF."""
    return "".join([page.extract_text() or "" for page in reader.pages])

def split_text_into_chunks(text, chunk_size=5000, overlap=100):
    """Divide o texto em partes menores."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def generate_summary_llm(text):
    """Gera um resumo do PDF utilizando a chain `summarize` do LangChain com prompt personalizado."""
    llm = Ollama(model="llama3.1")  # Define o modelo LLM

    # Prompt para a etapa MAP (resumo de trechos)
    map_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Você é um especialista em resumir textos acadêmicos."
        ),
        HumanMessagePromptTemplate.from_template(
            "Leia o trecho abaixo e faça um resumo conciso em português, sem equações matemáticas:\n\n{text}"
        )
    ])

    # Prompt para a etapa REDUCE (combinação dos resumos)
    reduce_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Agora, você vai combinar os resumos em um único texto coeso."
        ),
        HumanMessagePromptTemplate.from_template(
            "Aqui estão os resumos individuais:\n\n{text}\n\nPor favor, produza um resumo final em português:"
        )
    ])

    # Carregar a chain com os prompts personalizados
    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce", 
        map_prompt=map_prompt, 
        combine_prompt=reduce_prompt
    )
    summary = chain.run(text)  # Executa a chain com os chunks
    return summary

def create_pdf_with_summary(summary, output_path):
    """Cria um PDF contendo o resumo gerado."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    margin = 72
    y = height - margin

    # Divide o resumo em linhas de texto para caber na página
    lines = textwrap.wrap(summary, width=95)

    for line in lines:
        if y < margin:  # Cria uma nova página se necessário
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line)
        y -= 15

    c.save()

def find_abstract_end_page(reader):
    """Verifica se o PDF contém um abstract ou resumo."""
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if 'abstract' in text.lower() or 'resumo' in text.lower():
            return i + 1
    return None

def create_pdf_with_abstract_or_summary(input_path, output_directory):
    """Cria um resumo ou usa o abstract se presente no PDF."""
    with open(input_path, 'rb') as f:
        reader = PdfReader(f)
        filename_without_extension = os.path.splitext(os.path.basename(input_path))[0]
        sanitized_filename = sanitize_filename(filename_without_extension)
        output_filename = f"abstract_{sanitized_filename}.pdf"
        output_path = os.path.join(output_directory, output_filename)

        if os.path.exists(output_path):
            st.text(f"O resumo já existe para: {input_path}")
            return

        abstract_end_page = find_abstract_end_page(reader)
        writer = PdfWriter()

        if abstract_end_page:
            st.text(f"Abstract encontrado no arquivo: {input_path}")
            writer.add_page(reader.pages[0])
            writer.add_page(reader.pages[abstract_end_page - 1])
        else:
            st.text(f"Abstract não encontrado, gerando resumo para: {input_path}")
            pdf_text = extract_text_from_pdf(reader)
            summary = generate_summary_llm(pdf_text)

            temp_pdf_path = os.path.join(output_directory, "temp_summary.pdf")
            create_pdf_with_summary(summary, temp_pdf_path)

            with open(temp_pdf_path, 'rb') as temp_f:
                temp_reader = PdfReader(temp_f)
                writer.add_page(temp_reader.pages[0])
            os.remove(temp_pdf_path)

        with open(output_path, 'wb') as f_out:
            writer.write(f_out)

        st.text(f"Novo PDF criado: {output_path}")

def process_pdfs(input_directory, output_directory):
    """Processa todos os PDFs no diretório de entrada."""
    for filename in os.listdir(input_directory):
        if filename.endswith('.pdf'):
            input_path = os.path.join(input_directory, filename)
            create_pdf_with_abstract_or_summary(input_path, output_directory)

def load_pdfs_from_directory(directory):
    """Carrega e retorna uma lista de documentos do diretório."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                text = extract_text_from_pdf(reader)
                documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

def chat_with_model(prompt, model_name, docs, parameter):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    split_documents = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": parameter})

    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join([f"Documento: {doc.metadata['source']}\n{doc.page_content}" for doc in retrieved_docs])

    prompt_template = ChatPromptTemplate.from_template(
        "Você está auxiliando uma pesquisa. Responda com base nos documentos fornecidos.\n\n{context}\n\nPergunta: {input}"
    )
    llm = Ollama(model=model_name)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain.run({"context": context, "input": prompt})

# Interface Streamlit
st.title("ZCHAT")

input_directory = st.text_input("Diretório de entrada")
output_directory = st.text_input("Diretório de saída")
parameters = st.slider("Documentos recuperados", min_value=1, max_value=100)
model_name = st.selectbox("Escolha o modelo de IA", ["llama3.1", "phi 3", "mistral"])
prompt = st.text_area("Digite seu prompt")

if st.button("Resumir PDFs"):
    if input_directory and output_directory:
        process_pdfs(input_directory, output_directory)
    else:
        st.warning("Por favor, informe os diretórios.")

if st.button("Rodar IA"):
    if prompt:
        docs = load_pdfs_from_directory(output_directory)
        response = chat_with_model(prompt, model_name, docs, parameters)
        st.write("Resposta da IA:")
        st.text(response)
    else:
        st.warning("Por favor, digite um prompt.")

st.write(check_cuda())
st.info("Recomendamos enviar arquivos já resumidos para melhor desempenho.")

if st.button("Fechar Aplicação"):
    sys.exit()
