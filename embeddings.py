# embeddings.py
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

@st.cache_resource
def load_models():
    """Loads embedding model and cross-encoder reranker into memory once."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedding_model, reranker