# app.py
import streamlit as st
from config import DEFAULT_DOCS
from embeddings import load_models
from retriever import build_retrievers, hybrid_rerank_retrieval
from generator import generate_answer
import ui

# 1. Page Config
st.set_page_config(
    page_title="Hybrid RAG + Reranking Demo",
    page_icon="🚀",
    layout="wide"
)

# 2. Session State Setup
if "docs" not in st.session_state:
    st.session_state.docs = list(DEFAULT_DOCS)

# 3. Load Models & Indices
embedding_model, reranker = load_models()
bm25_retriever, vector_db = build_retrievers(st.session_state.docs, embedding_model)

# 4. Dynamic Document Handler
def handle_add_document():
    new_text = st.session_state.get("new_doc_text", "").strip()
    if new_text:
        st.session_state.docs.append(new_text)
        st.cache_resource.clear()
        st.session_state["new_doc_text"] = ""
        st.toast("✅ Knowledge Base updated successfully!", icon="🎉")

# 5. UI Layout Render
ui.render_header()
user_query, api_key_input, search_button = ui.render_inputs(handle_add_document)

# 6. Pipeline Execution
if search_button:
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving and reranking documents..."):
            top_candidates = hybrid_rerank_retrieval(
                query=user_query,
                bm25_retriever=bm25_retriever,
                vector_db=vector_db,
                reranker=reranker,
                top_k=3
            )

        if not top_candidates:
            st.error("No relevant context found in documents.")
        else:
            retrieved_context_str = "\n".join([f"- {doc.page_content}" for _, doc in top_candidates])
            
            with st.spinner("Generating answer with Gemini..."):
                answer = generate_answer(
                    user_query=user_query,
                    retrieved_context_str=retrieved_context_str,
                    api_key=api_key_input
                )

            ui.render_results(top_candidates, answer)