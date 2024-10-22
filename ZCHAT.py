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
from langchain.schema import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import torch
import textwrap
import time

# Funções auxiliares
def check_cuda():
    if torch.cuda.is_available():
        return "CUDA disponível. A aplicação rodará na GPU."
    else:
        return "CUDA não disponível. Recomendamos instalar o driver CUDA para melhor desempenho."

def sanitize_filename(filename):
    """Remove caracteres especiais e espaços do nome do arquivo."""
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    return filename.replace(' ', '_')

def load_pdfs_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text()
                # Criar um objeto Document para cada arquivo PDF com metadados
                documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

def process_pdfs(input_directory, output_directory):
    """Processa todos os PDFs no diretório de entrada."""
    for filename in os.listdir(input_directory):
        if filename.endswith('.pdf'):
            input_path = os.path.join(input_directory, filename)
            create_pdf_with_abstract_or_summary(input_path, output_directory)


def find_abstract_end_page(reader):
    """Localiza a última página do abstract ou resumo no PDF."""
    num_pages = len(reader.pages)
    for i in range(num_pages):
        try:
            page = reader.pages[i]
            text = page.extract_text() or ""
        except Exception as e:
            print(f"Erro ao extrair texto da página {i}: {e}")
            continue

        if 'abstract' in text.lower():
            return i + 1
        elif 'resumo'in text.lower():
            return 
        else: return None 

def create_pdf_with_abstract_or_summary(input_path, output_directory):
    try:
        reader = PdfReader(open(input_path, 'rb'))
    except Exception as e:
        print(f"Erro ao abrir o arquivo {input_path}: {e}")
        return

    writer = PdfWriter()
    filename_without_extension = os.path.splitext(os.path.basename(input_path))[0]
    sanitized_filename = sanitize_filename(filename_without_extension)
    output_filename = f"abstract_{sanitized_filename}.pdf"
    output_path = os.path.join(output_directory, output_filename)

    abstract_end_page = find_abstract_end_page(reader)

    if os.path.exists(output_path):
        return

    elif abstract_end_page:  # Se encontrar o abstract, adiciona as páginas
        st.text(f"Abstract encontrado no arquivo: {input_path}")
        writer.add_page(reader.pages[0])  # Primeira página
        writer.add_page(reader.pages[abstract_end_page])
    else:  # Se não encontrar o abstract, gera um resumo usando a LLM
        st.text(f"Abstract não encontrado, gerando resumo para: {input_path}")
        pdf_text = extract_text_from_pdf(reader)
        summary = generate_summary_llm(pdf_text)

        temp_pdf_path = os.path.join(output_directory, "temp_summary.pdf")
        create_pdf_with_summary(summary, temp_pdf_path)
        
        try:
            temp_reader = PdfReader(open(temp_pdf_path, 'rb'))
            writer.add_page(temp_reader.pages[0])
        except Exception as e:
            st.text(f"Erro ao ler o PDF temporário: {e}")
        finally:
            os.remove(temp_pdf_path)

    with open(output_path, 'wb') as f:
        writer.write(f)

    st.text(f"Novo PDF criado: {output_path}")

def generate_summary_llm(text):
    llm = Ollama(model="llama3.1")
    text_chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]  # Divide em blocos de 4000 caracteres
    summaries = []
    MAX_CHUNKS = 10  # Limite de chunks para processamento direto

    # Iteração sobre os chunks iniciais (primeiros 10 ou menos)
    for i, chunk in enumerate(text_chunks[:MAX_CHUNKS]):
        if i == 0:
            prompt = (
                "Você está escrevendo um resumo para um documento acadêmico. "
                "Resuma este documento sem incluir equações matemáticas, e cite o título do documento.\n\n"
                f"{chunk}"
            )
        else:
            prompt = (
                "Você está escrevendo um resumo para um documento acadêmico. "
                "Resuma este documento sem incluir equações matemáticas.\n\n"
                f"{chunk}"
            )

        summary = llm(prompt)
        summaries.append(summary.strip())

    # Caso haja mais de 10 chunks, faz um resumo adicional dos chunks excedentes
    if len(text_chunks) > MAX_CHUNKS:
        extra_chunks_text = "\n".join(text_chunks[MAX_CHUNKS:])  # Junta os chunks restantes
        prompt = (
            "Você está escrevendo um resumo para um documento acadêmico. "
            "Resuma o conteúdo a seguir, que é uma agregação das partes restantes do documento.\n\n"
            f"{extra_chunks_text}"
        )
        final_summary = llm(prompt)
        summaries.append(final_summary.strip())

    # Junta todos os resumos em uma única string e retorna
    return "\n".join(summaries)



def extract_text_from_pdf(reader):
    return "".join([page.extract_text() or "" for page in reader.pages])

def create_pdf_with_summary(text, output_path):
    """Cria um PDF contendo o resumo gerado."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter  # Dimensões da página
    margin = 72  # Margem padrão
    max_width = width - 2 * margin  # Largura máxima do texto
    y = height - margin  # Posição vertical inicial

    lines = textwrap.wrap(text, width=95)  # Ajusta automaticamente a largura das linhas

    for line in lines:
        c.drawString(margin, y, line)
        y -= 15  # Espaçamento entre linhas
        if y < margin:  # Quando atinge o final da página, cria uma nova
            c.showPage()
            y = height - margin

    c.save()

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

# Diretórios de entrada e saída
input_directory = st.text_input("Diretório de entrada (Coloque aqui documentos para serem resumidos)")
output_directory = st.text_input("Diretório de saída (Coloque aqui o diretório com os documentos resumidos, ou onde os documentos irão após o resumo)")
parameters = st.slider("Documentos recuperados pelo chatbot",min_value = 1,max_value=100)

# Seleção do modelo de IA
model_name = st.selectbox("Escolha o modelo de IA", ["llama3.1", "phi 3", "mistral"])

# Campo para o prompt
prompt = st.text_area("Digite seu prompt")

# Opção para resumir banco de dados
st.write("Esse resumo só deve ser feito uma vez, ou ao adicionar novos documentos ao banco de dados")
if st.button("Resumir banco de dados"):
    if input_directory and output_directory:
        process_pdfs(input_directory, output_directory)
    else:
        st.warning("Por favor, informe os diretórios.")

# Botão para rodar a IA com o prompt fornecido
if st.button("Rodar IA"):
    if prompt:
        docs = load_pdfs_from_directory(output_directory)
        response = chat_with_model(prompt, model_name, docs,parameters)
        st.write("Resposta da IA:")
        st.text(response)
    else:
        st.warning("Por favor, digite um prompt.")

# Exibir aviso sobre CUDA e desempenho
st.write(check_cuda())
st.info("Recomendamos enviar arquivos já resumidos para a IA para melhor desempenho.")

if st.button("Fechar Aplicação"):
    sys.exit()
