from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5:3b")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

parser = StrOutputParser()

chain = prompt | llm | parser

topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "Python",
    "Cloud Computing",
    "LangChain"
]

for topic in topics:
    print("=" * 50)
    print(chain.invoke({"topic": topic}))