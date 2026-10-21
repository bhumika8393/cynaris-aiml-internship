import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

prompts = [
    "Hello",
    "Explain Machine Learning in 100 words.",
    "Write a Python function for binary search with explanation.",
    "Summarize the following article. " * 40,
    "RAG Context: " + ("Artificial Intelligence is transforming cloud computing. " * 300)
]

print("=" * 60)
print("TOKEN COUNT AUDIT")
print("=" * 60)

for i, prompt in enumerate(prompts, start=1):
    tokens = len(enc.encode(prompt))

    input_cost = (tokens / 1_000_000) * 5

    print(f"\nPrompt {i}")
    print(f"Tokens : {tokens}")
    print(f"GPT-4o Estimated Input Cost : ${input_cost:.6f}")