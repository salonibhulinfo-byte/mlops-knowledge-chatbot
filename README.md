# MLOps Knowledge Chatbot 
# MLOps Knowledge Chatbot

A RAG-powered chatbot that helps you learn MLOps concepts - from Docker to model deployment to monitoring.

## What it does

Built this to learn production ML skills. You can ask questions like "How to containerize ML models?" or "What is MLflow?" and it searches through 25+ MLOps documents to give you answers.

## Documents included

**Research Papers (20):**
- MLOps fundamentals & best practices
- Model deployment strategies
- ML pipelines & automation
- Monitoring & testing approaches
- Feature stores & data management
- CI/CD for machine learning

**Tool Documentation (5):**
- Docker basics
- Kubernetes guide
- MLflow quickstart
- GitHub Actions for ML
- AWS SageMaker

## How I built it

1. Collected 25 MLOps documents (369 pages total)
2. Split into 1,573 searchable chunks
3. Created vector embeddings using sentence-transformers
4. Stored in FAISS vector database
5. Built RAG pipeline with LangChain
6. Added Streamlit chat interface

## Tech used

- Python
- LangChain
- HuggingFace (embeddings + LLM)
- FAISS (vector database)
- Streamlit (web UI)
- Transformers

## How to run it
```bash
# Clone and setup
git clone https://github.com/salonibhulinfo-byte/mlops-knowledge-chatbot.git
cd mlops-knowledge-chatbot
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Build vector database (first time only)
python build_vectorstore.py

# Run the chatbot
streamlit run app.py
```

## What I learned

- MLOps concepts (Docker, K8s, deployment, monitoring)
- How production ML differs from research
- Building domain-specific RAG applications
- Working with technical documentation
- Scaling RAG systems (1,573 chunks vs 1,177 in my first project)

## Things I'd improve

- Add source citations for answers
- Use better LLM (GPT-4) for clearer responses
- Add code examples from documentation
- Deploy online

Built while learning production ML skills after my MSc in Data Science at University of Greenwich. This is my second RAG project - first one was on AI research papers.