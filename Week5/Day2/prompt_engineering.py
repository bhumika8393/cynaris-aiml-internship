import requests
import subprocess
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
You are an AI tutor for beginners.
Explain concepts in simple English.
Keep answers short, accurate, and beginner-friendly.
If the question is about Ollama, explain the Ollama software used to run local Large Language Models (LLMs).
"""

prompts = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning.",
    "What is Prompt Engineering?",
    "Difference between AI and Machine Learning.",
    "Why use Ollama?"
]

comparison_questions = [
    "Explain LangChain.",
    "What is Retrieval Augmented Generation (RAG)?",
    "Why use local LLMs?"
]


def ask_model(model, prompt):
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        if response.status_code != 200:
            return f"HTTP Error {response.status_code}\n{response.text}"

        data = response.json()

        if "error" in data:
            return f"Ollama Error: {data['error']}"

        return data.get("response", "No response returned.")

    except Exception as e:
        return f"Error: {e}"


def stop_model(model):
    subprocess.run(
        ["ollama", "stop", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)


print("=" * 60)
print("PROMPT ENGINEERING & SYSTEM PROMPTS")
print("=" * 60)

# -----------------------------
# Part 1 - Test 5 Prompts
# -----------------------------
for i, prompt in enumerate(prompts, start=1):
    print(f"\nPrompt {i}: {prompt}\n")

    answer = ask_model("llama3.2:3b", prompt)
    print(answer)

    print("-" * 60)

# Unload llama before comparison
stop_model("llama3.2:3b")

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# -----------------------------
# Part 2 - Compare Models
# -----------------------------
for i, question in enumerate(comparison_questions, start=1):

    print(f"\nQuestion {i}: {question}")

    print("\n----- LLAMA 3.2 : 3B -----")
    llama_answer = ask_model("llama3.2:3b", question)
    print(llama_answer)

    # Free memory
    stop_model("llama3.2:3b")

    print("\n----- QWEN 2.5 : 3B -----")
    qwen_answer = ask_model("qwen2.5:3b", question)
    print(qwen_answer)

    # Free memory
    stop_model("qwen2.5:3b")

    print("\n" + "=" * 60)

print("\nProgram completed successfully.")