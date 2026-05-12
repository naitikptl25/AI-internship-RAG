"""
rag_engine.py
-------------
Retrieves relevant chunks from the vector store and uses
Groq (free tier) to generate grounded answers.

Model: llama-3.1-8b-instant via Groq API
- Completely free, no credit card needed
- Very fast inference (~500 tokens/sec)
- Get your free API key at: https://console.groq.com

Free tier limits: 14,400 requests/day, 500,000 tokens/day
"""

import time
import logging
from typing import List, Tuple, Dict

from groq import Groq

logger = logging.getLogger(__name__)

# Free Groq models (in order of preference)
# llama-3.1-8b-instant: fast, great for RAG Q&A
GENERATION_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions STRICTLY 
based on the provided context excerpts from the user's documents.

Rules:
1. ONLY use information present in the provided context.
2. If the answer is not in the context, say exactly:
   "I don't have enough information in the ingested documents to answer this."
3. Always mention the source document name(s) at the end of your answer.
4. Be concise and factual. Do not guess or infer beyond what the context states."""


class RAGEngine:
    """Retrieval-Augmented Generation using Groq (free tier LLaMA 3.1)."""

    def __init__(self, vector_store, api_key: str, top_k: int = 5):
        self.client       = Groq(api_key=api_key)
        self.vector_store = vector_store
        self.top_k        = top_k

    def _build_context(self, results: List[Tuple[Dict, float]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        lines = []
        for i, (chunk, score) in enumerate(results, 1):
            lines.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}\n")
        return "\n".join(lines)

    def _generate_with_retry(self, prompt: str, retries: int = 2) -> str:
        """Call Groq API with automatic retry on rate-limit errors."""
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=GENERATION_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.1,      # Low = more factual
                    max_tokens=600,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                # Rate limited — wait and retry
                if "429" in err and attempt < retries:
                    wait = 15 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s… (attempt {attempt+1})")
                    time.sleep(wait)
                # Daily quota exhausted
                elif "quota" in err.lower() or "rate_limit" in err.lower():
                    return (
                        "⚠️ Groq free-tier rate limit hit. "
                        "Please wait a moment and try again. "
                        "Limits reset every minute/day at console.groq.com"
                    )
                else:
                    raise

        return "⚠️ Failed to get a response after retries. Please try again shortly."

    def query(self, question: str) -> Dict:
        """Run full RAG: retrieve → build prompt → generate answer."""
        results = self.vector_store.search(question, top_k=self.top_k)

        if not results:
            return {
                "answer":       "No documents indexed yet. Please upload and build the index first.",
                "sources":      [],
                "context_used": "",
            }

        context = self._build_context(results)
        sources = list({chunk["source"] for chunk, _ in results})

        prompt = f"""Context from documents:
---
{context}
---

Question: {question}"""

        answer = self._generate_with_retry(prompt)

        return {
            "answer":       answer,
            "sources":      sources,
            "context_used": context,
        }
