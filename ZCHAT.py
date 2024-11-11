import os
import streamlit as st
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
import subprocess
import sys
import requests
import json
from tkinter import Tk, filedialog

CONFIG_FILE = 'directories_config.json'

def load_config():
    """Carrega os diretórios salvos de um arquivo JSON."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    return {"input_directory": "", "output_directory": ""}

def save_config(input_directory, output_directory):
    """Salva os diretórios em um arquivo JSON."""
    with open(CONFIG_FILE, 'w') as file:
        json.dump({"input_directory": input_directory, "output_directory": output_directory}, file)

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

def generate_summary_llm(text, model):
    """Gera um resumo do PDF utilizando a chain `summarize` do LangChain com prompt personalizado."""
    llm = Ollama(model=model)  # Define o modelo LLM

    # Prompt personalizado para a etapa MAP (resumo de trechos)
    map_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Você é um especialista em resumir textos acadêmicos."
        ),
        HumanMessagePromptTemplate.from_template(
            "Leia o seguinte trecho e faça um resumo conciso em português, sem equações matemáticas:\n\n{text}"
        )
    ])

    # Prompt personalizado para a etapa REDUCE (combinação dos resumos)
    reduce_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Agora, combine os resumos em um único texto coeso."
        ),
        HumanMessagePromptTemplate.from_template(
            "Aqui estão os resumos individuais:\n\n{text}\n\nPor favor, produza um resumo final em português:"
        )
    ])

    # Carrega a chain com os prompts personalizados
    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce", 
        map_prompt=map_prompt, 
        combine_prompt=reduce_prompt
    )

    # Divide o texto em chunks e cria objetos Document para cada chunk
    docs = [Document(page_content=chunk) for chunk in split_text_into_chunks(text)]

    # Executa a chain com os chunks
    summary = chain.run(docs)
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

def create_pdf_with_abstract_or_summary(input_path, output_directory, model):
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
            summary = generate_summary_llm(pdf_text, model)

            temp_pdf_path = os.path.join(output_directory, "temp_summary.pdf")
            create_pdf_with_summary(summary, temp_pdf_path)

            with open(temp_pdf_path, 'rb') as temp_f:
                temp_reader = PdfReader(temp_f)
                writer.add_page(temp_reader.pages[0])
            os.remove(temp_pdf_path)

        with open(output_path, 'wb') as f_out:
            writer.write(f_out)

        st.text(f"Novo PDF criado: {output_path}")

def process_pdfs(input_directory, output_directory, model):
    if check_model_exists(model_name) == False:
        download_model(model_name)
    for filename in os.listdir(input_directory):
        if filename.endswith('.pdf'):
            input_path = os.path.join(input_directory, filename)
            create_pdf_with_abstract_or_summary(input_path, output_directory, model)

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

def check_model_exists(model_name: str) -> bool:
    try:
        response = requests.get(f"http://localhost:11434/models/{model_name}")
        return response.status_code == 200
    except requests.ConnectionError:
        print("Erro: Não foi possível conectar ao serviço Ollama.")
        return False

def download_model(model_name: str):
    try:
        st.text(f"Iniciando o download do modelo: {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        st.text(f"Download do modelo {model_name} concluído.")
    except subprocess.CalledProcessError as e:
        st.text(f"Erro ao baixar o modelo {model_name}: {e}")
# Armazenar histórico de mensagens
chat_history = []

def chat_with_model(prompt, model_name, docs, parameter, chunk_size, chunk_overlap):
    if not check_model_exists(model_name):
        download_model(model_name)

    # Carregar embeddings e dividir os documentos
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": parameter})

    # Recuperar documentos relevantes
    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join([f"Documento: {doc.metadata['source']}\n{doc.page_content}" for doc in retrieved_docs])

    # Limitar o histórico a um número razoável de interações, por exemplo, as últimas 5
    limited_history = chat_history[-10:]  # Ajuste o número conforme necessário
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in limited_history])

    # Criar o prompt com histórico e contexto
    full_prompt = f"Você está auxiliando uma pesquisa. Responda a pergunta do usuário com base no contexto apresentado e no histórico de covnersas: \n\nHistórico de conversas:\n{formatted_history}\n\nContexto dos documentos:\n{context}\n\nPergunta do usuário: {prompt}"

    # Criar e rodar a cadeia de LLM
    llm = Ollama(model=model_name)
    llm_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template(full_prompt))
    response = llm_chain.run({"context": context, "input": prompt})

    # Adicionar a nova interação ao histórico
    chat_history.append({"role": "user", "content": prompt})
    chat_history.append({"role": "assistant", "content": response})

    return response


# Interface Streamlit
responses = ''
st.title("ZCHAT")
config = load_config()
with st.sidebar: 
    input_directory = st.text_input("Diretório de entrada", value=config.get("input_directory", ""))
    output_directory = st.text_input("Diretório de saída", value=config.get("output_directory", ""))

    if st.button("Salvar diretórios"):
        save_config(input_directory, output_directory)

    model_name = st.selectbox("Escolha o modelo de IA", ["llama3.1", "phi3", "mistral"])

    if st.button("Resumir PDFs"):
        
        if input_directory and output_directory:
            process_pdfs(input_directory, output_directory, model_name)
        else:
            st.warning("Por favor, informe os diretórios.")

    parameters = st.slider("Documentos recuperados", min_value=1, max_value=100, value=10)

    with st.expander("Configurações Avançadas"): 
        chunk_size = st.slider("Chunk size", min_value=512, max_value=4096, value=1024)
        chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=400,value=100)


messages = st.container(height=300)
prompt = st.chat_input("Digite seu prompt")
if prompt: 
    messages.chat_message("usuário").write(prompt)
    docs = load_pdfs_from_directory(output_directory)
    response = chat_with_model(prompt, model_name, docs, parameters, chunk_size,chunk_overlap)
    st.write("Resposta da IA:")
    messages.chat_message("ZCHAT").write(response)

if st.button("Fechar Aplicação"):
    sys.exit()
