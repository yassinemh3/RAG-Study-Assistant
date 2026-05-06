"""
app.py
------
Streamlit UI for the Smart Study Assistant RAG system.

Run with: streamlit run app.py
"""

import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedder
from src.vector_store import build_vector_store, load_vector_store
from src.generator import answer_question

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark background */
  .stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* Chat message bubbles */
  .user-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    font-size: 0.95rem;
    line-height: 1.5;
  }

  .assistant-bubble {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e8e8f0;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 85%;
    backdrop-filter: blur(10px);
    font-size: 0.95rem;
    line-height: 1.6;
  }

  /* Header */
  .main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
  }

  .main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
  }

  .main-header p {
    color: rgba(255,255,255,0.5);
    font-size: 1rem;
  }

  /* Status badge */
  .status-ready {
    background: rgba(39, 174, 96, 0.15);
    border: 1px solid rgba(39, 174, 96, 0.4);
    color: #27ae60;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-block;
    margin-top: 0.5rem;
  }

  .status-not-ready {
    background: rgba(231, 76, 60, 0.15);
    border: 1px solid rgba(231, 76, 60, 0.4);
    color: #e74c3c;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-block;
    margin-top: 0.5rem;
  }

  /* Source expander */
  .source-tag {
    background: rgba(102,126,234,0.15);
    border: 1px solid rgba(102,126,234,0.3);
    color: #a0aec0;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    display: inline-block;
    margin: 2px 3px;
  }

  /* Input box */
  .stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
    width: 100%;
  }

  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102,126,234,0.4);
  }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []     # chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None


# ---------------------------------------------------------------------------
# Helper: load embedder once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_embedder():
    return get_embedder()


# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 Study Materials")
    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Drop your lecture notes, textbooks, or study guides here.",
    )

    if uploaded_files:
        os.makedirs("docs", exist_ok=True)
        for uploaded_file in uploaded_files:
            save_path = os.path.join("docs", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to docs/")

    st.markdown("---")

    # Ingestion button
    if st.button("Index Documents", use_container_width=True):
        doc_files = [f for f in os.listdir("docs") if f.endswith((".pdf", ".txt"))] if os.path.exists("docs") else []
        if not doc_files:
            st.error("No documents found in docs/. Please upload files first.")
        else:
            with st.spinner("Indexing documents..."):
                try:
                    embedder = cached_embedder()
                    docs = load_documents("docs")
                    chunks = chunk_documents(docs)
                    vs = build_vector_store(chunks, embedder)
                    st.session_state.vector_store = vs
                    st.session_state.embedder = embedder
                    st.success(f"Indexed {len(docs)} file(s) | {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    # Load existing index button
    st.markdown("---")
    if st.button("Load Existing Index", use_container_width=True):
        with st.spinner("Loading index from disk..."):
            try:
                embedder = cached_embedder()
                vs = load_vector_store(embedder)
                st.session_state.vector_store = vs
                st.session_state.embedder = embedder
                st.success("Index loaded successfully!")
            except FileNotFoundError:
                st.error("No saved index found. Please index documents first.")
            except Exception as e:
                st.error(f"Failed to load index: {e}")

    # Status indicator
    st.markdown("---")
    st.markdown("**Status**")
    if st.session_state.vector_store is not None:
        st.markdown('<span class="status-ready">Ready to answer questions</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-not-ready">No index loaded</span>', unsafe_allow_html=True)

    # Current docs list
    if os.path.exists("docs"):
        doc_files = [f for f in os.listdir("docs") if f.endswith((".pdf", ".txt"))]
        if doc_files:
            st.markdown("---")
            st.markdown("**Indexed files:**")
            for f in doc_files:
                st.markdown(f"- `{f}`")

    # Clear chat
    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
  <h1>Smart Study Assistant</h1>
  <p>Ask questions about your study materials — get grounded, cited answers.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; color: rgba(255,255,255,0.3); padding: 3rem 0;">
          <div style="font-size: 3rem;">📖</div>
          <p style="margin-top: 1rem;">Upload and index your study materials, then ask a question below.</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Show sources if available
            if "sources" in msg and msg["sources"]:
                with st.expander("View sources used"):
                    for i, doc in enumerate(msg["sources"], 1):
                        src = doc.metadata.get("source", "unknown")
                        st.markdown(f'<span class="source-tag">Source {i}: {src}</span>', unsafe_allow_html=True)
                        st.caption(doc.page_content[:300] + "...")


# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------
st.markdown("---")

with st.form(key="question_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="Ask a question",
            placeholder="e.g. What is the difference between overfitting and underfitting?",
            label_visibility="collapsed",
        )
    with col2:
        submit = st.form_submit_button("Ask", use_container_width=True)

if submit and user_input.strip():
    if st.session_state.vector_store is None:
        st.warning("Please index your documents first using the sidebar.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate answer
        with st.spinner("Thinking..."):
            try:
                result = answer_question(
                    question=user_input,
                    vector_store=st.session_state.vector_store,
                    k=4,
                    return_sources=True,
                )
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as e:
                answer = f"Sorry, I encountered an error: {e}"
                sources = []

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        st.rerun()
