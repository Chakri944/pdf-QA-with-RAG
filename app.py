import streamlit as st
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import re

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

st.title("PDF Asker")
st.write("Ask questions from your documents")

user_doc = st.file_uploader("Upload a document", type='pdf')

# only process PDF when a NEW file is uploaded
if user_doc and st.session_state.get("last_file") != user_doc.name:
    st.session_state["last_file"] = user_doc.name
    
    with open("temp.pdf", "wb") as f:
        f.write(user_doc.getvalue())
    
    loader = PyPDFLoader(file_path='temp.pdf')
    documents = loader.load()
    split = text_splitter.split_documents(documents)
    st.session_state["split"] = split
    
    sample_context = "\n\n".join([chunk.page_content for chunk in split[0:3]])
    prompt = f"""
    Generate exactly 3 questions based on the context below.
    Return only the questions numbered 1. 2. 3. with no intro text.
    
    Context:
    {sample_context}
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    questions = re.split(r'\d+\.', result)
    questions = [q.strip() for q in questions if q.strip()]
    st.session_state["questions"] = questions

# show suggested questions
if "questions" in st.session_state:
    st.markdown("**Suggested Questions:**")
    for question in st.session_state["questions"]:
        if st.button(question):
            st.session_state["selected"] = question

# text input
user_query = st.text_input("Or type your own question", 
                           value=st.session_state.get("selected", ""))

if st.button("Send"):
    if user_query and "split" in st.session_state:
        with st.spinner("Fetching answer..."):
            try:
                client_db = chromadb.Client()
                client_db.delete_collection("temp_collection")
            except:
                pass
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(
                st.session_state["split"],
                embeddings,
                collection_name="temp_collection"
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            relevant_chunks = retriever.invoke(user_query)
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            
            prompt = f"""
            You are a helpful assistant.
            Answer the question based only on the context below.
            
            Context:
            {context}
            
            Question:
            {user_query}
            """
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(response.choices[0].message.content)
    else:
        st.warning("Please upload a PDF and enter a question")