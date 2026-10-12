"""
Week 6 Day 1
LangChain Fundamentals - Chains, Memory and Agents
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# Prompt Template
# -------------------------------

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple words."
)

parser = StrOutputParser()

print("=== Prompt Template ===")

topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "LangChain",
    "Ollama"
]

for topic in topics:
    formatted_prompt = prompt.format(topic=topic)
    print("\nPrompt:")
    print(formatted_prompt)

print("\n====================================")

# -------------------------------
# Conversation Memory Simulation
# -------------------------------

print("\n=== Conversation Memory ===")

conversation_history = []

messages = [
    "Hello",
    "What is LangChain?",
    "What is Memory?",
    "Explain Agents",
    "Thank you"
]

for message in messages:
    conversation_history.append(message)

print("\nConversation History")

for i, msg in enumerate(conversation_history, start=1):
    print(f"{i}. {msg}")

print("\n====================================")

# -------------------------------
# Tool 1
# -------------------------------

def calculator(expression):
    return eval(expression)

# -------------------------------
# Tool 2
# -------------------------------

def web_search(query):
    return f"Simulated web search result for: {query}"

print("\n=== Agent Simulation ===")

print(web_search("LangChain"))
print(calculator("15 + 25"))
print(web_search("Ollama LLM"))

print("\nAgent executed all tasks successfully.")