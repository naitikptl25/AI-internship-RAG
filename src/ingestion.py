"""
ingestion.py
------------
Handles loading and reading documents.
Supports two modes:
  1. From uploaded file bytes (Streamlit UploadedFile objects) — primary mode
  2. From a local folder path — fallback / CLI mode

Supported formats: PDF, TXT, CSV
"""

import io
import csv
import logging
from pathlib import Path
from typing import List, Dict

import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Per-format readers (work on raw bytes) ─────────────────────────────────────

def read_txt_bytes(data: bytes) -> str:
    """Decode plain text bytes to string."""
    return data.decode("utf-8", errors="replace")


def read_pdf_bytes(data: bytes) -> str:
    """Extract text from all pages of a PDF given as raw bytes."""
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def read_csv_bytes(data: bytes) -> str:
    """
    Convert CSV bytes into readable text.
    Each row becomes: "Field1: val1 | Field2: val2"
    """
    text = data.decode("utf-8", errors="replace")
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        line = " | ".join(f"{k}: {v}" for k, v in row.items())
        rows.append(line)
    return "\n".join(rows)


# ── Format dispatcher ──────────────────────────────────────────────────────────

READERS = {
    ".txt":  read_txt_bytes,
    ".pdf":  read_pdf_bytes,
    ".csv":  read_csv_bytes,
}


# ── Mode 1: Streamlit uploaded files (primary) ─────────────────────────────────

def ingest_uploaded_files(uploaded_files) -> List[Dict]:
    """
    Process a list of Streamlit UploadedFile objects directly in memory.
    No temp files written to disk.

    Args:
        uploaded_files: List of st.file_uploader results.

    Returns:
        List of dicts with keys: 'source', 'content', 'file_type'
    """
    documents = []

    for uf in uploaded_files:
        ext = Path(uf.name).suffix.lower()
        if ext not in READERS:
            logger.warning(f"Skipping unsupported file: {uf.name}")
            continue

        try:
            data = uf.read()           # Read bytes from the uploaded file
            content = READERS[ext](data)

            if content.strip():
                documents.append({
                    "source":    uf.name,
                    "content":   content,
                    "file_type": ext.lstrip("."),
                })
                logger.info(f"Loaded upload: {uf.name} ({len(content)} chars)")
        except Exception as e:
            logger.error(f"Failed to read {uf.name}: {e}")

    logger.info(f"Total uploaded documents processed: {len(documents)}")
    return documents


# ── Mode 2: Local folder (CLI / fallback) ──────────────────────────────────────

def ingest_documents(documents_dir: str) -> List[Dict]:
    """
    Walk a local directory and load all supported files.

    Args:
        documents_dir: Path to folder containing PDF/TXT/CSV files.

    Returns:
        List of dicts with keys: 'source', 'content', 'file_type'
    """
    documents = []
    docs_path = Path(documents_dir)

    if not docs_path.exists():
        logger.warning(f"Documents directory not found: {documents_dir}")
        return documents

    for file_path in sorted(docs_path.iterdir()):
        ext = file_path.suffix.lower()
        if ext not in READERS:
            continue

        try:
            data = file_path.read_bytes()
            content = READERS[ext](data)

            if content.strip():
                documents.append({
                    "source":    file_path.name,
                    "content":   content,
                    "file_type": ext.lstrip("."),
                })
                logger.info(f"Loaded: {file_path.name} ({len(content)} chars)")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
