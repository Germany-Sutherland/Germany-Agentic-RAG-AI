import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="Germany Job Market AI Demo", layout="wide")

st.title("🇩🇪 Germany Job Market – AI Skills Q&A")
st.markdown("""
This free, open-source demo answers questions about the German job market, 
AI skills demand, and hiring trends.  
Built with **Hugging Face Transformers** and **Streamlit** – no API key needed.
""")

# Load model only once
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_qa_pipeline()

# Context – replace with real dataset later
context = """
Germany has a high demand for AI, machine learning, cloud computing, data science,
and cybersecurity professionals. Major hiring hubs include Berlin, Munich, and Frankfurt.
Top industries hiring tech talent: automotive, finance, healthcare, and manufacturing.
Skilled labor shortages are acute in engineering, IT, and green energy sectors.
"""

# User input
question = st.text_input("Ask a question about the German job market:")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Analyzing..."):
            try:
                result = qa_pipeline(question=question, context=context)
                st.subheader("Answer:")
                st.success(result['answer'])
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please type a question before clicking 'Get Answer'.")

st.markdown("---")
st.caption("Made with ❤️ in Streamlit & Hugging Face – Free & Open Source.")
