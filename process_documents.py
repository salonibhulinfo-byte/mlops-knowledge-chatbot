from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents():
    documents = []
    data_folder = "data"

    print("Loading MLOps documents...")
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            print(f"Loading: {filename}")

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

def split_documents(documents):
    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    return chunks

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    
    print("\n" + "="*50)
    print("Example chunk:")
    print("="*50)
    print(chunks[0].page_content[:500])