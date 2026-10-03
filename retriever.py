# retriever.py
import streamlit as st
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

@st.cache_resource(show_spinner=False)
def build_retrievers(doc_list, embedding_model):
    """Builds BM25 and FAISS vector index."""
    documents = [Document(page_content=text) for text in doc_list if text.strip()]
    if not documents:
        return None, None
    
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 5

    vector_db = FAISS.from_documents(documents, embedding_model)
    return bm25, vector_db


def hybrid_rerank_retrieval(query, bm25_retriever, vector_db, reranker, top_k=3):
    """Performs BM25 + FAISS hybrid search followed by Cross-Encoder reranking."""
    if not vector_db or not bm25_retriever:
        return []

    bm25_docs = bm25_retriever.invoke(query)
    dense_docs = vector_db.as_retriever(search_kwargs={"k": 5}).invoke(query)

    # Deduplicate candidate documents
    candidate_pool = {doc.page_content: doc for doc in (bm25_docs + dense_docs)}
    candidates = list(candidate_pool.values())

    if not candidates:
        return []

    # Cross-Encoder Reranking
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored_candidates = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return scored_candidates[:top_k]