"""
Week 6 Day 2
LangChain Memory & Conversation History
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Prompt Template
# -----------------------------

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question in simple words:\n{question}"
)

parser = StrOutputParser()

print("=" * 50)
print("LANGCHAIN PROMPT TEMPLATE")
print("=" * 50)

questions = [
    "What is Artificial Intelligence?",
    "What is LangChain?",
    "What is Conversation Memory?",
    "What is an Agent?",
    "Why is memory useful?"
]

for i, question in enumerate(questions, start=1):
    formatted = prompt.format(question=question)
    print(f"\nInput {i}")
    print(formatted)

print("\n" + "=" * 50)

# -----------------------------
# Conversation Memory Simulation
# -----------------------------

print("Conversation Memory")

conversation_history = []

messages = [
    "Hello!",
    "Explain LangChain.",
    "What is ConversationBufferMemory?",
    "Can you remember previous messages?",
    "Thank you!"
]

for message in messages:
    conversation_history.append(message)

print("\nStored Conversation")

for i, msg in enumerate(conversation_history, start=1):
    print(f"{i}. {msg}")

print("\nConversation maintained across 5 turns.")

print("\n" + "=" * 50)

# -----------------------------
# Agent Tools
# -----------------------------

def calculator(expression):
    return eval(expression)

def web_search(query):
    return f"Simulated search result for '{query}'"

print("Agent Tasks\n")

print(web_search("LangChain Memory"))
print(calculator("250 + 375"))
print(web_search("Conversation History"))

print("\nAgent completed all tasks successfully.")