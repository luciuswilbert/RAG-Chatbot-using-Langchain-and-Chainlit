import fitz  # PyMuPDF
import os
import requests

from dotenv import load_dotenv

load_dotenv()

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

    embeddings = []
    for idx, chunk in enumerate(chunks):
        print(f"Embedding chunk {idx+1}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

    # Optionally, print the length of embeddings and one example
    print(f"Total embeddings: {len(embeddings)}")
    print(f"First embedding vector (first 10 values): {embeddings[0][:10]}")


