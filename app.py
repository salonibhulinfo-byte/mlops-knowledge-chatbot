import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Page config
st.set_page_config(page_title="MLOps Knowledge Assistant", page_icon="🚀")

@st.cache_resource
def load_rag_chain():
    """Load vector store and create RAG chain (cached)"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    
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
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """Answer the question based on the MLOps context below. Keep it concise.

Context: {context}

Question: {question}

Answer: """
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Title
st.title("🚀 MLOps Knowledge Assistant")
st.markdown("Ask questions about MLOps, deployment, monitoring, and production ML!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load RAG chain
with st.spinner("Loading AI model..."):
    chain = load_rag_chain()

# Chat input
if prompt := st.chat_input("Ask about MLOps..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching MLOps knowledge..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot helps you learn MLOps concepts using RAG.
    
    **Topics covered:**
    - Docker & Kubernetes
    - Model deployment
    - MLflow & experiment tracking
    - CI/CD for ML
    - Monitoring & testing
    - Feature stores
    - Cloud platforms (AWS, GCP)
    
    **Built with:**
    - LangChain
    - HuggingFace
    - FAISS
    - Streamlit
    
    **Documents:** 25+ MLOps guides & papers
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()