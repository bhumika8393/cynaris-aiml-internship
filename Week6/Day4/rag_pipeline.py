print("Step 1: Importing modules...")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

print("Step 2: Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Step 3: Opening ChromaDB...")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("Step 4: Loading Ollama model...")

llm = OllamaLLM(model="qwen2.5:3b")

print("Step 5: Ready!")

query = input("Ask a question: ")

print("Searching...")

docs = db.similarity_search(query, k=3)

context = "\n\n".join(doc.page_content for doc in docs)

print("\nRetrieved Context:\n")
print(context)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

print("\nGenerating answer...\n")

response = llm.invoke(prompt)

print("\nLLM Answer:\n")
print(response)