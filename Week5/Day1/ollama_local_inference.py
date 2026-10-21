"""
Week 5 Day 1
Running LLMs Locally with Ollama
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer clearly, accurately, and concisely."
)

PROMPTS = [
    "What is Artificial Intelligence?",
    "Explain LangChain.",
    "What is Ollama?",
    "What is Retrieval Augmented Generation?",
    "Why should developers use local LLMs?"
]


def ask_model(model_name, prompt):
    payload = {
        "model": model_name,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        return response.json()["response"]
    return f"Error: {response.status_code}"


print("=" * 60)
print("LLAMA 3.2 : 3B")
print("=" * 60)

for prompt in PROMPTS:
    print(f"\nPrompt: {prompt}")
    print(ask_model("llama3.2:3b", prompt))

print("\n" + "=" * 60)
print("QWEN 2.5 : 3B")
print("=" * 60)

questions = [
    "Explain Machine Learning.",
    "What is Python?",
    "Why is Git useful?"
]

for question in questions:
    print(f"\nQuestion: {question}")

    print("\nLlama3.2")
    print(ask_model("llama3.2:3b", question))

    print("\nQwen2.5")
    print(ask_model("qwen2.5:3b", question))

    print("-" * 60)