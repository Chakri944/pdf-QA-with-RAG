# PDF Q&A with RAG

## What it does
Upload any PDF and ask questions about it. 
The app finds the most relevant sections and answers accurately using AI.

## How it works
1. PDF is split into chunks
2. Chunks are converted to vectors using HuggingFace embeddings
3. Stored in ChromaDB vector database
4. User question is matched against chunks
5. Relevant chunks + question sent to Groq LLM
6. Answer returned from document context only

## Tech Stack
- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API (Llama 3.3 70B)
- Streamlit
- Deployed on Hugging Face Spaces

## How to run locally
pip install -r requirements.txt
streamlit run app.py