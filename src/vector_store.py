"""
vector_store.py
---------------
Generates embeddings using Google Gemini (gemini-embedding-001)
and stores/retrieves them using FAISS.

Uses the new google-genai SDK (google-genai package).
Get a free API key at: https://aistudio.google.com/apikey
"""

import pickle
import logging
import time
from typing import List, Dict, Tuple

import faiss
import numpy as np
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"

class VectorStore:
    """FAISS vector store backed by Gemini embeddings."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.chunks: List[Dict] = []

        # Detect real embedding dimension dynamically on first init
        # gemini-embedding-001 returns 3072 dims (not 768)
        test = self._embed_texts(["test"], task_type="RETRIEVAL_DOCUMENT")
        dim = test.shape[1]
        self.index = faiss.IndexFlatIP(dim)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-10)

    def _embed_texts(self, texts: List[str], task_type: str) -> np.ndarray:
        """
        Embed a list of texts with Gemini.
        task_type: 'RETRIEVAL_DOCUMENT' for indexing, 'RETRIEVAL_QUERY' for queries.
        Retries once on rate-limit (429) with a 10-second back-off.
        """
        vectors = []
        for text in texts:
            for attempt in range(2):   # Up to 2 attempts
                try:
                    response = self.client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=text,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    vectors.append(response.embeddings[0].values)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt == 0:
                        logger.warning("Rate limited — waiting 10 s…")
                        time.sleep(10)
                    else:
                        raise

        arr = np.array(vectors, dtype=np.float32)
        return self._normalise(arr)

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict]) -> None:
        """Embed every chunk and add to the FAISS index."""
        logger.info(f"Embedding {len(chunks)} chunks…")
        texts = [c["text"] for c in chunks]

        # Process one at a time — safe for free-tier rate limits
        all_vecs = self._embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        self.index.add(all_vecs)
        self.chunks.extend(chunks)
        logger.info(f"Index now has {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Return top-k chunks most similar to the query."""
        if self.index.ntotal == 0:
            return []

        q_vec = self._embed_texts([query], task_type="RETRIEVAL_QUERY")
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path + ".index")
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str) -> None:
        self.index  = faiss.read_index(path + ".index")
        with open(path + ".meta", "rb") as f:
            self.chunks = pickle.load(f)

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0
