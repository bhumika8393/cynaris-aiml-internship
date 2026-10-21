import requests

questions = [
    "What is Artificial Intelligence?",
    "Explain semantic search.",
    "What is Retrieval-Augmented Generation?"
]

models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

for model in models:
    print("\n" + "=" * 60)
    print(f"MODEL: {model}")
    print("=" * 60)

    for question in questions:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": question,
                "stream": False
            }
        )

        print(f"\nQuestion: {question}")
        print(response.json()["response"])