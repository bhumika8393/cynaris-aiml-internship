from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

qa_pairs = [
    ("What is AI?", "Artificial Intelligence enables machines to perform human-like tasks."),
    ("What is Python?", "Python is a programming language."),
    ("Explain Machine Learning", "ML is a subset of AI."),
    ("What is ChromaDB?", "Vector database for embeddings."),
    ("What is Ollama?", "Runs LLMs locally."),
]

questions = [q for q, _ in qa_pairs]
answers = [a for _, a in qa_pairs]

embeddings = model.encode(questions)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = input("Enter your question: ")

query_embedding = model.encode(query)

scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]

best = np.argmax(scores)

print("\nSimilarity:", round(scores[best], 3))

if scores[best] >= 0.92:
    print("\nCache Hit")
    print(answers[best])
else:
    print("\nCache Miss")
    print("Call the LLM here.")