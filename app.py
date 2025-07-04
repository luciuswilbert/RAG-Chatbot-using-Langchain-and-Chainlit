import fitz  # PyMuPDF
import os
import requests

from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

class LMStudioEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # texts: List[str]
        return [get_embedding(t) for t in texts]
    def embed_query(self, text):
        return get_embedding(text)

LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL")  

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    """
    Splits text into chunks with optional overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(chunk):
    payload = {
        "model": os.getenv("EMBEDDING_MODEL_NAME"),
        "input": chunk
    }
    response = requests.post(f"{LMSTUDIO_API_URL}/embeddings", json=payload)
    response.raise_for_status()
    embedding = response.json()["data"][0]["embedding"]
    return embedding

def query_faiss(faiss_path, query, k=4):
    embedding_fn = LMStudioEmbeddings()
    faiss_db = FAISS.load_local(
        faiss_path,
        embeddings=embedding_fn,
        allow_dangerous_deserialization=True
    )
    query_embedding = embedding_fn.embed_query(query)
    results = faiss_db.similarity_search_by_vector(query_embedding, k=k)
    return results  

def generate_llm_answer_langchain(context, user_query):
    """
    Uses LangChain AzureChatOpenAI to generate an answer from retrieved context and user query.
    """
    # These come from .env
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")

    # Initialize the LangChain LLM
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        deployment_name=azure_deployment,
        api_version=api_version,
        temperature=0.1,
    )

    # Prepare messages
    system = SystemMessage(content="You are a helpful assistant that answers based only on the given context.")
    user = HumanMessage(content=f"Context:\n{context}\n\nUser Question: {user_query}\n\nAnswer using only the given context.")

    response = llm.invoke([system, user])
    return response.content.strip()

if __name__ == "__main__":
    pdf_path = "Resume - Lucius Wilbert Tjoa.pdf"  
    extracted_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(extracted_text)

    embedding_fn = LMStudioEmbeddings()

    documents = [Document(page_content=chunk) for chunk in chunks]

    faiss_db = FAISS.from_documents(
        documents=documents,
        embedding=embedding_fn
    )

    faiss_db.save_local("my_faiss_index")
    user_query = input("Enter your question: ")
    results = query_faiss("my_faiss_index", user_query)

    context = "\n\n".join([doc.page_content for doc in results])
    prompt = (
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        "Answer using only the given context."
    )

    llm_answer = generate_llm_answer_langchain(context, user_query)
    print("\nLLM Answer:\n", llm_answer)



