# MLOps Knowledge Chatbot

A chatbot that teaches MLOps concepts by answering questions about Docker, Kubernetes, model deployment, and production machine learning.

## Why I built this

After building my first RAG chatbot on AI research papers, I realized I understood the theory but not how to actually deploy ML models to production. So I built this second chatbot specifically to learn MLOps - the practical side of getting models into the real world.

## What it does

Ask it questions like:
- "How do I containerize my ML model?"
- "What's the difference between MLflow and Kubeflow?"
- "How do I monitor models in production?"

It searches through 25 MLOps documents and gives you answers based on what it finds.

## Documents I used

I collected two types of content:

**20 Research Papers on:**
- MLOps fundamentals
- Model deployment strategies
- ML pipelines and automation
- Monitoring and testing
- Feature stores and data versioning

**5 Tool Guides on:**
- Docker
- Kubernetes
- MLflow
- GitHub Actions for ML
- AWS SageMaker

Total: 369 pages → 1,573 searchable chunks

## How it works

Same approach as my first project:
1. Load all the PDFs
2. Break them into chunks
3. Turn chunks into vector embeddings
4. Store in FAISS database
5. When you ask a question, find relevant chunks
6. Feed chunks to an LLM to generate an answer
7. Show answer in Streamlit chat interface

## Tech stack

- Python
- LangChain (for the RAG pipeline)
- HuggingFace (embeddings and LLM)
- FAISS (vector search)
- Streamlit (web interface)

## How to run it
```bash
git clone https://github.com/salonibhulinfo-byte/mlops-knowledge-chatbot.git
cd mlops-knowledge-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python build_vectorstore.py
streamlit run app.py
```

## What I learned

- Production ML is very different from notebook ML
- MLOps covers way more than just deployment
- Docker and Kubernetes are essential for ML engineers
- Building a second RAG project was much faster (knew what worked)
- Bigger isn't always better - 1,573 chunks takes longer to search


