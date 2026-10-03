import os
import streamlit as st
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types

# Set Page Config
st.set_page_config(
    page_title="Hybrid RAG + Reranking Demo",
    page_icon="🚀",
    layout="wide"
)

# ----------------------------------------------------------------------
# 1. INITIALIZE MODELS & CACHE
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads embedding model and cross-encoder reranker into memory once."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedding_model, reranker

embedding_model, reranker = load_models()

# Initialize session state for document store
if "docs" not in st.session_state:
    st.session_state.docs = [
        "Error 404: Web server failed to locate requested resource path /api/v1/auth.",
        "System maintenance is scheduled every Sunday at 02:00 AM UTC.",
        "The client-side React frontend handles user state management via Redux Toolkit.",
        "For user authentication issues, reset your OAuth tokens or check error code 404.",
        "To deploy updates to Cloudflare Pages, run the build script npm run deploy.",
        "Database backup strategy includes daily automated snapshots at midnight."
    ]

@st.cache_resource(show_spinner=False)
def build_retrievers(doc_list):
    """Builds BM25 and FAISS vector index."""
    documents = [Document(page_content=text) for text in doc_list if text.strip()]
    if not documents:
        return None, None
    
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 5

    vector_db = FAISS.from_documents(documents, embedding_model)
    return bm25, vector_db

bm25_retriever, vector_db = build_retrievers(st.session_state.docs)

# ----------------------------------------------------------------------
# 2. RETRIEVAL & RERANKING
# ----------------------------------------------------------------------
def hybrid_rerank_retrieval(query, top_k=3):
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

# ----------------------------------------------------------------------
# 3. STREAMLIT UI LAYOUT
# ----------------------------------------------------------------------
st.title("🚀 Hybrid RAG + Cross-Encoder Reranking Showcase")
st.markdown(
    "Demonstration of **Sparse Keyword (BM25)** + **Dense Vector (FAISS)** search, "
    "re-scored with a **Cross-Encoder Reranker** and powered by **Gemini 2.5 Flash**."
)

st.divider()

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
        help="Provide your API key or set GEMINI_API_KEY as an environment secret on Streamlit Cloud."
    )
    search_button = st.button("🔍 Search & Generate Answer", type="primary")

with col2:
    st.subheader("📥 Add Dynamic Knowledge")
    new_doc_input = st.text_area("Add New Document Chunk", height=100)
    if st.button("Update Knowledge Base"):
        if new_doc_input.strip():
            st.session_state.docs.append(new_doc_input.strip())
            st.cache_resource.clear()  # Rebuild indices with new doc
            st.success("Successfully added chunk and updated indices!")
        else:
            st.warning("Please enter valid text.")

st.divider()

# ----------------------------------------------------------------------
# 4. EXECUTION & OUTPUT
# ----------------------------------------------------------------------
if search_button:
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving and reranking documents..."):
            top_candidates = hybrid_rerank_retrieval(user_query, top_k=3)

        if not top_candidates:
            st.error("No relevant context found in documents.")
        else:
            # Display Retrieval Results
            res_col1, res_col2 = st.columns([1, 1])

            retrieved_context_str = ""
            debug_md = "### 🔍 Retrieved & Reranked Chunks\n\n"

            for rank, (score, doc) in enumerate(top_candidates, 1):
                retrieved_context_str += f"- {doc.page_content}\n"
                debug_md += f"**Rank {rank}** (Score: `{score:.4f}`)\n> {doc.page_content}\n\n"

            with res_col2:
                st.markdown(debug_md)

            # Generate Answer via Gemini API
            api_key = api_key_input.strip() or os.environ.get("GEMINI_API_KEY")
            
            if not api_key:
                with res_col1:
                    st.error("⚠️ Please provide a Gemini API Key or set GEMINI_API_KEY in Streamlit Secrets.")
            else:
                with st.spinner("Generating answer with Gemini..."):
                    try:
                        system_instruction = (
                            "You are an enterprise AI assistant. Answer the user question using ONLY "
                            "the provided context. If the answer cannot be found in the context, say "
                            "'I cannot find the answer in the provided documents.'"
                        )
                        prompt = f"Context:\n{retrieved_context_str}\n\nUser Question: {user_query}"

                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.2,
                            )
                        )

                        with res_col1:
                            st.subheader("🤖 Generated Answer")
                            if hasattr(response, 'text') and response.text:
                                st.write(response.text)
                            else:
                                st.warning("Gemini returned an empty response or it was filtered by safety settings.")

                    except Exception as e:
                        with res_col1:
                            st.error(f"⚠️ Gemini API Error: {str(e)}")