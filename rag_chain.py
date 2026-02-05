from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_vectorstore():
    """Load the existing vector store"""
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector store loaded!")
    return vectorstore

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore):
    """Create the RAG chain with local HuggingFace model"""
    print("\nLoading language model...")
    
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ Model loaded!")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """Answer the question based on the MLOps context below. Keep it concise.

Context: {context}

Question: {question}

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG chain ready!")
    return chain

def ask_question(chain, question):
    """Ask a question and get an answer"""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    answer = chain.invoke(question)
    
    print(f"\nAnswer: {answer}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    chain = create_rag_chain(vectorstore)
    
    questions = [
        "How to containerize a machine learning model?",
        "What is MLOps?",
        "How to monitor ML models in production?"
    ]
    
    for question in questions:
        ask_question(chain, question)