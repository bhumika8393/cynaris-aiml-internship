from langchain_core.messages import HumanMessage, AIMessage

conversation = []

def chat(user_input, ai_response):
    conversation.append(HumanMessage(content=user_input))
    conversation.append(AIMessage(content=ai_response))

chat("Hi", "Hello! How can I help you?")
chat("My name is Bhumika.", "Nice to meet you, Bhumika.")
chat("I study Information Science Engineering.", "That's great!")
chat("I like AI and Machine Learning.", "Excellent choice.")
chat("What is my name?", "Your name is Bhumika.")

print("=" * 50)
print("Conversation History")
print("=" * 50)

for msg in conversation:
    if isinstance(msg, HumanMessage):
        print(f"Human: {msg.content}")
    else:
        print(f"AI    : {msg.content}")