"""
chunker.py
----------
Splits raw document text into overlapping chunks for better retrieval.
Overlap helps preserve context across chunk boundaries.
"""

from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text:       Raw document text.
        chunk_size: Number of characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Only keep non-empty chunks
        if chunk:
            chunks.append(chunk)

        # Advance by (chunk_size - overlap) so adjacent chunks share context
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Chunk all loaded documents and attach metadata to each chunk.

    Args:
        documents:  List of document dicts from ingestion.py
        chunk_size: Characters per chunk.
        overlap:    Overlap between consecutive chunks.

    Returns:
        List of chunk dicts with keys: 'text', 'source', 'chunk_id'
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "source": doc["source"],  # Retain original filename for attribution
                    "chunk_id": f"{doc['source']}_chunk_{idx}",
                }
            )

    return all_chunks
