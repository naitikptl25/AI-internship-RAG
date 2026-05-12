"""
app.py
------
Streamlit web UI for the Agentic RAG System (Gemini-powered).
Users upload documents directly in the browser — no local folder needed.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
GEMINI_API_KEY = "AIzaSyCV3lho3j7fiNjLD76OpXNMASLWEgNyjNU"
GROQ_API_KEY   = "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"

# Allow imports from /src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingestion import ingest_uploaded_files
from chunker import chunk_documents
from vector_store import VectorStore
from rag_engine import RAGEngine

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #0d0f14; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #2a2d36;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #161a22;
    border: 2px dashed #2a2d36;
    border-radius: 12px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover { border-color: #38bdf8; }

/* File pill badges */
.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e3a4a;
    color: #38bdf8;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px 4px 3px 0;
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid #2a4a5a;
}

/* Chat cards */
.question-card {
    background: #161a22;
    border: 1px solid #2a2d36;
    border-radius: 12px 12px 12px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.answer-card {
    background: #0f1923;
    border-left: 3px solid #38bdf8;
    border-radius: 0 12px 12px 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.5rem;
    font-size: 0.96rem;
    line-height: 1.75;
}
.source-tag {
    display: inline-block;
    background: #1e3a4a;
    color: #38bdf8;
    border-radius: 5px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 4px 4px 0 0;
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid #2a4a5a;
}
.context-box {
    background: #0d1117;
    border: 1px solid #2a2d36;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: #94a3b8;
    white-space: pre-wrap;
    max-height: 280px;
    overflow-y: auto;
}
.metric-box {
    background: #161a22;
    border: 1px solid #2a2d36;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-num   { font-size: 1.7rem; font-weight: 600; color: #38bdf8; }
.metric-label { font-size: 0.72rem; color: #64748b; margin-top: 3px; }

.empty-state {
    background: #161a22;
    border: 1px dashed #2a2d36;
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
    color: #475569;
}

h1, h2, h3 { color: #f1f5f9 !important; }

/* Buttons */
.stButton > button {
    background: #38bdf8;
    color: #0d0f14;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    transition: 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #0ea5e9; transform: translateY(-1px); }

/* Text input */
.stTextInput > div > div > input {
    background: #161a22 !important;
    border: 1px solid #2a2d36 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    padding: 0.7rem 1rem;
}
.stTextInput > div > div > input:focus { border-color: #38bdf8 !important; }

/* Divider */
hr { border-color: #2a2d36; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "vector_store": None,
    "rag_engine":   None,
    "history":      [],
    "doc_stats":    {"docs": 0, "chunks": 0},
    "indexed_names": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind RAG")
    st.markdown("---")

    # API key status
    if GEMINI_API_KEY:
        st.success("✅ Gemini API key loaded")
    else:
        st.error("❌ GEMINI_API_KEY missing in .env\nGet a free key at aistudio.google.com/apikey")
    api_key = GEMINI_API_KEY

    st.markdown("---")

    # ── File uploader ──────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Documents")
    st.caption("Supports PDF, TXT, CSV — multiple files at once")

    uploaded_files = st.file_uploader(
        label="Drop files here",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show uploaded file names as pills
    if uploaded_files:
        icons = {".pdf": "📕", ".txt": "📄", ".csv": "📊"}
        pills_html = ""
        for uf in uploaded_files:
            ext = os.path.splitext(uf.name)[1].lower()
            icon = icons.get(ext, "📎")
            pills_html += f'<span class="file-pill">{icon} {uf.name}</span>'
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Chunking settings ──────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    chunk_size    = st.slider("Chunk Size (chars)",    200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap (chars)",  50,  200, 100, 25)
    top_k         = st.slider("Retrieved Chunks (top-k)", 1, 10, 5)

    st.markdown("---")

    # ── Build Index button ─────────────────────────────────────────────────────
    if st.button("🔨 Build Index from Uploads", use_container_width=True):
        if not api_key or not GROQ_API_KEY:
            st.error("Add both GEMINI_API_KEY and GROQ_API_KEY to your .env file.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Reading → Chunking → Embedding… this may take a moment"):
                try:
                    # Step 1 — Ingest directly from uploaded bytes (no disk writes)
                    docs = ingest_uploaded_files(uploaded_files)

                    if not docs:
                        st.error("No readable content found in the uploaded files.")
                    else:
                        # Step 2 — Chunk
                        chunks = chunk_documents(docs, chunk_size, chunk_overlap)

                        # Step 3 — Embed + index with Gemini
                        vs = VectorStore(api_key=api_key)
                        vs.add_chunks(chunks)

                        # Step 4 — Save to session
                        st.session_state.vector_store  = vs
                        st.session_state.rag_engine    = RAGEngine(vs, GROQ_API_KEY, top_k)
                        st.session_state.doc_stats     = {"docs": len(docs), "chunks": len(chunks)}
                        st.session_state.indexed_names = [d["source"] for d in docs]
                        st.session_state.history       = []  # Reset chat on new index

                        st.success(f"✅ {len(docs)} doc(s) → {len(chunks)} chunks indexed!")

                except Exception as e:
                    import traceback
                    st.error(f"❌ {type(e).__name__}: {e}")
                    st.code(traceback.format_exc(), language="text")

    # ── Index stats ────────────────────────────────────────────────────────────
    if st.session_state.vector_store:
        st.markdown("### 📊 Index Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-num">{st.session_state.doc_stats["docs"]}</div>'
                f'<div class="metric-label">Documents</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-num">{st.session_state.doc_stats["chunks"]}</div>'
                f'<div class="metric-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )

        # List indexed files
        st.markdown("**Indexed files:**")
        for name in st.session_state.indexed_names:
            ext = os.path.splitext(name)[1].lower()
            icon = {".pdf": "📕", ".txt": "📄", ".csv": "📊"}.get(ext, "📎")
            st.markdown(f"- {icon} `{name}`")

        st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown(
        '<p style="font-size:0.72rem;color:#475569;margin-top:1rem;">'
        'AI Internship Assignment · Agentic RAG System<br>'
        'Embeddings: Gemini · LLM: Groq LLaMA 3.1 (both free)</p>',
        unsafe_allow_html=True,
    )

# ── Main Panel ─────────────────────────────────────────────────────────────────
st.markdown("# 🧠 DocMind")
st.markdown(
    '<p style="color:#64748b;margin-top:-12px;margin-bottom:1.5rem;">'
    'Upload your documents in the sidebar → Build Index → Ask anything below.</p>',
    unsafe_allow_html=True,
)

# ── Query input area ───────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")

with st.container():
    question = st.text_input(
        "question",
        value=prefill,
        placeholder="Ask a question about your uploaded documents...",
        label_visibility="collapsed",
    )

    col_ask, col_ctx = st.columns([3, 2])
    with col_ask:
        ask_clicked = st.button("🔍 Ask", use_container_width=True)
    with col_ctx:
        show_context = st.checkbox("Show retrieved chunks", value=False)

# ── Process query ──────────────────────────────────────────────────────────────
if ask_clicked and question.strip():
    if not st.session_state.rag_engine:
        st.warning("⬅️ Upload documents and click **Build Index** first.")
    else:
        st.session_state.rag_engine.top_k = top_k  # Respect slider change
        with st.spinner("Searching documents and generating answer…"):
            result = st.session_state.rag_engine.query(question.strip())

        st.session_state.history.append({
            "question": question.strip(),
            "answer":   result["answer"],
            "sources":  result["sources"],
            "context":  result["context_used"],
        })
        st.rerun()  # Refresh to show answer at top of history

# ── Chat history ───────────────────────────────────────────────────────────────
st.markdown("---")

if st.session_state.history:
    for entry in reversed(st.session_state.history):
        # Question bubble
        st.markdown(
            f'<div class="question-card">🙋 {entry["question"]}</div>',
            unsafe_allow_html=True,
        )

        # Answer bubble + source tags
        sources_html = "".join(
            f'<span class="source-tag">📄 {s}</span>' for s in entry["sources"]
        )
        st.markdown(
            f'<div class="answer-card">{entry["answer"]}'
            f'<br><br>{sources_html}</div>',
            unsafe_allow_html=True,
        )

        # Optional raw context
        if show_context and entry["context"]:
            with st.expander("🔍 Retrieved context chunks"):
                st.markdown(
                    f'<div class="context-box">{entry["context"]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

else:
    # Empty state illustration
    if st.session_state.vector_store:
        placeholder = "Index is ready — ask your first question above!"
        icon = "💬"
    else:
        placeholder = "Upload your documents in the sidebar and click Build Index to get started."
        icon = "📂"

    st.markdown(
        f'<div class="empty-state"><div style="font-size:2.5rem;margin-bottom:0.8rem">{icon}</div>'
        f'<div>{placeholder}</div></div>',
        unsafe_allow_html=True,
    )
