from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Opening ChromaDB...")

db = Chroma(
    persist_directory="../Day4/chroma_db",
    embedding_function=embeddings
)

print("Loading Ollama...")

llm = OllamaLLM(model="qwen2.5:3b")

conversation = []

print("\nDocument Chatbot Ready!")
print("Type 'exit' to quit.\n")
while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    history = "\n".join(conversation)

    prompt = f"""
You are a helpful AI assistant.

Conversation History:
{history}

Document Context:
{context}

User Question:
{question}

Answer using the document context whenever possible.
"""

    answer = llm.invoke(prompt)

    conversation.append(f"User: {question}")
    conversation.append(f"Assistant: {answer}")

    print("\nAssistant:\n")
    print(answer)
    print("-" * 60)