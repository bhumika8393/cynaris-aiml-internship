import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(name="documents")

# 20 sample documents
documents = [
    "Python is a popular programming language.",
    "Machine learning enables computers to learn from data.",
    "Artificial Intelligence is transforming industries.",
    "LangChain helps build LLM applications.",
    "Ollama allows running LLMs locally.",
    "ChromaDB stores vector embeddings.",
    "Embeddings represent semantic meaning.",
    "Semantic search finds related content.",
    "Cosine similarity measures vector closeness.",
    "Deep learning uses neural networks.",
    "RAG combines retrieval with generation.",
    "Large Language Models understand natural language.",
    "Data science uses statistics and programming.",
    "NLP processes human language.",
    "Python supports many AI libraries.",
    "Vector databases improve search quality.",
    "CrewAI builds AI agents.",
    "LangGraph creates agent workflows.",
    "MLflow tracks machine learning experiments.",
    "Ragas evaluates Retrieval Augmented Generation."
]

ids = [str(i) for i in range(len(documents))]

embeddings = model.encode(documents).tolist()

metadatas = []

for i in range(len(documents)):
    metadatas.append(
        {
            "category": "AI" if i < 10 else "ML",
            "index": i
        }
    )

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Added 20 documents.\n")

query = "What is semantic search?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("Similarity Search Results:\n")

for doc in results["documents"][0]:
    print("-", doc)

print("\nMetadata Filter:\n")

filtered = collection.get(
    where={"category": "AI"}
)

for doc in filtered["documents"][:5]:
    print(doc)