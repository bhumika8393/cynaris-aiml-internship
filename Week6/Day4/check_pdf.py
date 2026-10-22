from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/ai_notes.pdf")
docs = loader.load()

print("Number of pages:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n--- Page {i+1} ---")
    print("Characters:", len(doc.page_content))
    print(doc.page_content[:300])