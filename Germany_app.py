# app.py — Agentic RAG Demo for Germany Job Market
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from textwrap import shorten

# Set Streamlit page config
st.set_page_config(page_title="Agentic RAG — Germany Job Market", layout="wide")

st.title("🇩🇪 Agentic RAG — Germany Job Market (Demo)")
st.markdown(
    """
    Retrieval-Augmented Generation demo using FAISS + Hugging Face.
    Answers questions about the German job market without any paid API keys.
    """
)

# Load models once and cache them
@st.cache_resource(show_spinner=False)
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return embed_model, qa_model

embed_model, qa_pipeline = load_models()

# Small demo knowledge base (replace with real job postings later)
documents = [
    "Germany has strong demand for AI, data science, and software engineering roles, "
    "with major hubs in Berlin, Munich, and Hamburg. Companies hire ML engineers, "
    "data scientists, MLOps, and cloud engineers.",
    
    "The green energy sector in Germany is expanding. Renewable energy, wind, solar, "
    "and EV manufacturing create jobs in engineering, analytics, and project management.",
    
    "Frankfurt is Germany’s finance hub; fintech and quantitative roles are concentrated "
    "there and in Berlin. Demand for cloud, cybersecurity, and compliance skills is high.",
    
    "Healthcare and digital health are growing in Germany: telemedicine, healthcare data "
    "analysts, and health IT engineers are in demand.",
    
    "Common skills in demand: Python, SQL, cloud platforms (AWS, GCP, Azure), "
    "Docker/Kubernetes, machine learning frameworks, NLP, and large language models."
]

# Build FAISS index
@st.cache_resource(show_spinner=False)
def build_faiss_index(docs):
    embeddings = embed_model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

index = build_faiss_index(documents)

# Retrieve top-k documents
def retrieve_docs(query, k=2):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    return [documents[i] for i in indices[0]]

# Streamlit UI
query = st.text_input("Ask a question about the German job market:")

col1, col2 = st.columns([2, 1])

with col2:
    top_k = st.selectbox("Docs to retrieve (k)", [1, 2, 3], index=1)
    show_agent_step = st.checkbox("Show agent reasoning", value=True)

with col1:
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            hits = retrieve_docs(query, k=top_k)
            context = "\n\n".join(hits)
            
            agent_decision = "Answer concisely based on retrieved context."
            if "where" in query.lower():
                agent_decision = "Focus on location and city information."
            elif "skill" in query.lower():
                agent_decision = "Focus on relevant skills and technologies."
            
            if show_agent_step:
                st.info(f"Agent reasoning: {agent_decision}")
            
            try:
                result = qa_pipeline({"question": query, "context": context})
                answer = result.get("answer", "No answer found.")
                score = result.get("score", 0)
                
                st.subheader("Answer")
                st.success(shorten(answer, width=800))
                st.caption(f"Confidence: {score:.2f}")
                
                st.markdown("**Retrieved Documents:**")
                for i, doc in enumerate(hits, 1):
                    st.write(f"**Doc {i}:** {shorten(doc, 400)}")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with FAISS, Hugging Face Transformers, and Streamlit — Free & Open-Source.")
