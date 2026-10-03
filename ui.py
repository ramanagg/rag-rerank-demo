# ui.py
import streamlit as st

def render_header():
    st.title("🚀 Hybrid RAG + Cross-Encoder Reranking Showcase")
    st.markdown(
        "Demonstration of **Sparse Keyword (BM25)** + **Dense Vector (FAISS)** search, "
        "re-scored with a **Cross-Encoder Reranker** and powered by **Gemini 2.5 Flash**."
    )
    st.divider()

def render_inputs(on_add_doc_callback):
    col1, col2 = st.columns([2, 1])

    with col1:
        user_query = st.text_area(
            "Ask a Question",
            placeholder="e.g., How do I fix error 404 during authentication?",
            height=100
        )
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Provide your API key or set GEMINI_API_KEY in Streamlit Secrets."
        )
        search_button = st.button("🔍 Search & Generate Answer", type="primary")

    with col2:
        st.subheader("📥 Add Dynamic Knowledge")
        st.text_area(
            "Add New Document Chunk", 
            height=100, 
            key="new_doc_text",
            placeholder="Type new information to index..."
        )
        st.button("Update Knowledge Base", on_click=on_add_doc_callback)

        with st.expander(f"📚 View All Indexed Chunks ({len(st.session_state.docs)})"):
            for i, doc in enumerate(st.session_state.docs, 1):
                st.markdown(f"**{i}.** {doc}")

    st.divider()
    return user_query, api_key_input, search_button

def render_results(top_candidates, answer):
    res_col1, res_col2 = st.columns([1, 1])

    debug_md = "### 🔍 Retrieved & Reranked Chunks\n\n"
    for rank, (score, doc) in enumerate(top_candidates, 1):
        debug_md += f"**Rank {rank}** (Score: `{score:.4f}`)\n> {doc.page_content}\n\n"

    with res_col2:
        st.markdown(debug_md)

    with res_col1:
        st.subheader("🤖 Generated Answer")
        st.write(answer)