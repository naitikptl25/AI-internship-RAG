# 🧠 DocMind — Agentic RAG System

> An AI-powered document chatbot that answers questions **strictly from your uploaded documents** — grounded answers, cited sources, zero hallucinations.

---

## 📐 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  Sidebar: Upload docs, API key status, settings, Build Index     │
│  Main:    Query input, chat history, source citations            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
             ┌───────────▼────────────┐
             │       RAG Engine       │  ← rag_engine.py
             │  retrieve → prompt →   │
             │  Groq LLaMA 3.1 answer │
             └───────────┬────────────┘
                         │
             ┌───────────▼────────────┐
             │      Vector Store      │  ← vector_store.py
             │  FAISS + Gemini        │
             │  Embeddings (3072-dim) │
             └───────────┬────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │         Document Pipeline           │
        │  ingestion.py  →  chunker.py        │
        │  PDF / TXT / CSV  →  500-char chunks│
        └─────────────────────────────────────┘
```

### Data Flow

1. **Upload** — User drags & drops PDF / TXT / CSV files in the browser
2. **Ingest** — Files are read in-memory (no disk writes) via `ingestion.py`
3. **Chunk** — Text split into 500-char overlapping windows (100-char overlap)
4. **Embed** — Each chunk embedded via `gemini-embedding-001` → 3072-dim vectors
5. **Index** — Vectors stored in FAISS `IndexFlatIP` (cosine similarity)
6. **Query** — User question embedded → top-k chunks retrieved from FAISS
7. **Generate** — Context + question sent to Groq `llama-3.1-8b-instant`
8. **Respond** — Grounded answer with source file citations shown in UI

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| **UI** | Streamlit | Rapid web UI, file upload, chat interface |
| **Embeddings** | Gemini `gemini-embedding-001` | Free, high-quality 3072-dim embeddings |
| **Vector DB** | FAISS `IndexFlatIP` | In-memory, no infra needed, exact cosine search |
| **LLM** | Groq `llama-3.1-8b-instant` | Free, 14,400 req/day, ~500 tok/sec, great for RAG |
| **PDF parsing** | PyPDF2 | Lightweight, no external dependencies |
| **Language** | Python 3.10+ | Per assignment requirement |

---

## 📁 Project Structure

```
rag_system/
├── app.py                    ← Streamlit UI (entry point)
├── requirements.txt          ← Python dependencies
├── .env.example              ← API key template
├── README.md                 ← This file
├── WRITEUP.md                ← Architecture decisions & scaling
│
├── src/
│   ├── ingestion.py          ← Load PDF, TXT, CSV from uploads
│   ├── chunker.py            ← Fixed-size overlapping chunking
│   ├── vector_store.py       ← FAISS + Gemini embeddings
│   └── rag_engine.py         ← Groq LLaMA retrieval + generation
│
└── documents/                ← Sample documents (upload via UI)
    ├── sop_employee_onboarding.txt
    ├── product_documentation_cloudstore.txt
    ├── refund_returns_policy.txt
    └── product_catalog.csv
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/your-username/docmind-rag.git
cd docmind-rag
pip install -r requirements.txt
```

### 2. Get Free API Keys (no credit card for either)

**Gemini** (for embeddings):
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with Google → Create API Key

**Groq** (for LLM generation):
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → API Keys → Create API Key

### 3. Configure `.env`

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=AIza-your-gemini-key-here
GROQ_API_KEY=gsk_your-groq-key-here
```

### 4. Run the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 5. Use the App

1. **Upload documents** — drag & drop PDF/TXT/CSV files in the sidebar
2. **Click "🔨 Build Index"** — ingests, chunks, and embeds your documents
3. **Ask a question** — type in the main panel and click "🔍 Ask"
4. **See the answer** — response shown with source file citations

---

## 🧪 Example Queries

Use the sample documents in the `documents/` folder to test:

| Query | Source Document |
|---|---|
| What are the onboarding steps for a new employee? | `sop_employee_onboarding.txt` |
| How do I reset my CloudStore Pro password? | `product_documentation_cloudstore.txt` |
| What is the refund policy for monthly subscriptions? | `refund_returns_policy.txt` |
| Which products are currently out of stock? | `product_catalog.csv` |
| What are the system requirements for CloudStore Pro? | `product_documentation_cloudstore.txt` |
| How long does a physical product return take to process? | `refund_returns_policy.txt` |
| What happens on Day 1 of employee onboarding? | `sop_employee_onboarding.txt` |

---

## ⚙️ Configuration

| Setting | Default | Description |
|---|---|---|
| Chunk Size | 500 chars | Size of each text chunk |
| Chunk Overlap | 100 chars | Shared context between adjacent chunks |
| Top-k | 5 | Number of chunks retrieved per query |
| Embedding Model | `gemini-embedding-001` | Gemini embedding model (free) |
| Generation Model | `llama-3.1-8b-instant` | Groq LLaMA model (free) |

---

## 🔑 Free Tier Limits

| Service | Limit |
|---|---|
| Gemini Embeddings | 1,500 requests/day |
| Groq LLaMA 3.1 | 14,400 requests/day · 500,000 tokens/day |

---

## 📦 Dependencies

```
google-genai      — Gemini embeddings (free)
groq              — LLaMA 3.1 generation via Groq (free)
faiss-cpu         — Vector similarity search
PyPDF2            — PDF text extraction
streamlit         — Web UI
numpy             — Vector math
python-dotenv     — Load .env file
```

---

## 📝 Write-Up

See [`WRITEUP.md`](WRITEUP.md) for architecture decisions, limitations, and scaling plan.
