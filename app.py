import fitz  # PyMuPDF
import os
import requests

from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


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


def chunk_text(text, chunk_size=800, overlap=200):
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
    print("API response:", response.json())  # Add this line
    embedding = response.json()["data"][0]["embedding"]
    return embedding

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



