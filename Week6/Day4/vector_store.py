from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("documents/ai_notes.pdf")
documents = loader.load()

print("Pages loaded:", len(documents))
print("Characters on first page:", len(documents[0].page_content))
print(documents[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="chroma_db"
)

print(f"Stored {len(docs)} document chunks.")