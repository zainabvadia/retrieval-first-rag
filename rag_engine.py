"""
rag_engine.py
Core retrieval-augmented generation engine.

Design principle: retrieval quality is the bottleneck in most RAG systems,
not the language model. This engine is built to make retrieval quality
inspectable and tunable, not just a black box wrapped around an LLM call.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Loads a lightweight open-source embedding model.
        all-MiniLM-L6-v2 is used here deliberately: it's small (~80MB),
        fast on CPU, and strong enough for semantic similarity on
        short-to-medium length text chunks, a reasonable production
        default before reaching for something heavier.
        """
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
        """
        Splits text into overlapping word-based chunks.

        Overlap matters: without it, a fact that spans a chunk boundary
        can become unretrievable because neither chunk fully contains it.
        This is a small detail that has an outsized effect on real-world
        retrieval accuracy.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def ingest(self, documents: dict[str, str]):
        """
        documents: dict of {source_name: full_text}
        Chunks each document and embeds every chunk, storing the source
        so retrieved results can be traced back to where they came from.
        """
        self.chunks = []
        for source, text in documents.items():
            for chunk in self.chunk_text(text):
                self.chunks.append({"source": source, "text": chunk})

        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Returns the top_k most semantically relevant chunks to the query,
        along with a similarity score, ranked highest first.

        Uses cosine similarity via normalized dot product, since embeddings
        were normalized at encode time. This keeps retrieval fast and
        dependency-light (no external vector database required for a
        small-to-medium document set).
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("No documents ingested yet. Call ingest() first.")

        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "text": self.chunks[idx]["text"],
                "score": float(scores[idx]),
            })
        return results

    def save_index(self, path: str = "index.json"):
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_index(self, path: str = "index.json"):
        with open(path, "r") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.embeddings = np.array(data["embeddings"])