from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_and_split_documents():
    """Load PDFs and split into chunks"""
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
    
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks

def create_vectorstore(chunks):
    """Create embeddings and store in FAISS"""
    print("\nCreating embeddings (this may take a few minutes)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore.save_local("vectorstore")
    print("\n✅ Vector store created and saved!")
    
    return vectorstore

def  test_search(vectorstore):
    """Test if search works"""
    print("\n" + "="*50)
    print("Testing search...")
    print("="*50)
    
    query = "How to deploy a machine learning model?"
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"\nQuery: '{query}'")
    print(f"\nTop result:\n")
    print(results[0].page_content[:300] + "...")

if __name__ == "__main__":
    chunks = load_and_split_documents()
    vectorstore = create_vectorstore(chunks)
    test_search(vectorstore)
    