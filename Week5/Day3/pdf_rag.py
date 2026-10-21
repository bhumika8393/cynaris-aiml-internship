import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import requests

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./pdf_db")

collection = client.get_or_create_collection(
    name="pdf_docs"
)

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])

embeddings = model.encode(chunks).tolist()

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

question = "Summarize this PDF."

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context = "\n".join(results["documents"][0])

prompt = f"""
Use the context below to answer.

Context:
{context}

Question:
{question}
"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
)

print("\nRetrieved Chunks:\n")

for chunk in results["documents"][0]:
    print("-"*50)
    print(chunk)

print("\nLLM Answer:\n")

print(response.json()["response"])