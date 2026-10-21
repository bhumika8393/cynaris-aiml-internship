import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import requests

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("qa_pdf")

# Read PDF
reader = PdfReader("sample.pdf")

text = ""
for page in reader.pages:
    if page.extract_text():
        text += page.extract_text()

# Split into chunks
chunk_size = 500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Generate embeddings
embeddings = model.encode(chunks).tolist()

ids = [str(i) for i in range(len(chunks))]

try:
    collection.delete(ids=ids)
except:
    pass

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("Local Q&A Bot Ready!")

while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    results = collection.query(
        query_embeddings=[model.encode(question).tolist()],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    system_prompt = (
        "You are a helpful AI assistant. "
        "Answer only using the provided context. "
        "If the answer is not available, reply 'Not found in the document.'"
    )

    prompt = f"""
System:
{system_prompt}

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

    print("\nAnswer:\n")
    print(response.json()["response"])