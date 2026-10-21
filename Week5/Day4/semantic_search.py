import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("semantic_docs")

documents = [
    "Python is a programming language.",
    "Artificial Intelligence enables machines to think.",
    "Machine Learning learns patterns from data.",
    "Deep Learning uses neural networks.",
    "Natural Language Processing works with text.",
    "Computer Vision processes images.",
    "ChromaDB is a vector database.",
    "Embeddings represent semantic meaning.",
    "Cosine similarity compares vectors.",
    "RAG combines retrieval with generation.",
    "LangChain helps build LLM applications.",
    "LangGraph manages AI workflows.",
    "CrewAI creates AI agents.",
    "MLflow tracks machine learning experiments.",
    "Ollama runs LLMs locally.",
    "Semantic search retrieves similar content.",
    "Vector databases store embeddings.",
    "ChatGPT is a conversational AI model.",
    "Transformers power modern language models.",
    "AI improves automation and decision making."
]

ids = [str(i) for i in range(len(documents))]
embeddings = model.encode(documents).tolist()

metadatas = [
    {"category": "AI" if i < 10 else "Tools"}
    for i in range(len(documents))
]

# Clear previous data if it exists
try:
    collection.delete(ids=ids)
except:
    pass

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("20 documents added successfully.\n")

query = "Explain semantic search"

results = collection.query(
    query_embeddings=[model.encode(query).tolist()],
    n_results=3
)

print("Top 3 Similar Documents:\n")

for i, doc in enumerate(results["documents"][0], 1):
    print(f"{i}. {doc}")

print("\nMetadata Filter (AI):\n")

filtered = collection.get(where={"category": "AI"})

for doc in filtered["documents"][:5]:
    print("-", doc)