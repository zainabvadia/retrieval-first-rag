"""
rag_engine.py
Core retrieval-augmented generation engine.

Design principle: retrieval quality is the bottleneck in most RAG systems,
not the language model. This engine is built to make retrieval quality
inspectable and tunable, not just a black box wrapped around an LLM call.
"""

import re
import json
import numpy as np


class SentenceTransformerEmbedder:
    """
    Default embedder: a lightweight open-source sentence embedding model.
    all-MiniLM-L6-v2 is used deliberately: it's small (~80MB), fast on CPU,
    and strong enough for semantic similarity on short-to-medium chunks.
    The import is lazy so the rest of the project (and the test suite)
    works without downloading the model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


class HashingEmbedder:
    """
    Tiny dependency-free embedder: hashes words into a fixed-size vector.
    It only captures word overlap, not meaning, so it is NOT a substitute
    for a real embedding model. It exists so tests and CI can run fast,
    offline and deterministically, and so the engine's logic (chunking,
    ranking, thresholds) can be tested separately from the model.
    """

    _STOP = {"the", "a", "an", "is", "are", "do", "does", "to", "be", "of", "in", "on",
             "for", "and", "or", "what", "when", "where", "how", "i", "my", "can", "need"}

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9$]+", text.lower()):
            if w in self._STOP:
                continue
            h = int.from_bytes(__import__("hashlib").md5(w.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


class RAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedder=None):
        """
        embedder: any object with encode(list[str]) -> normalized np.ndarray.
        Defaults to the sentence-transformers model, which keeps retrieval
        and embedding separate so each can be swapped, tested and evaluated
        on its own.
        """
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder(model_name)
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
        self.embeddings = self.embedder.encode(texts)

    def retrieve(self, query: str, top_k: int = 3, min_score: float | None = None) -> list[dict]:
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

        query_embedding = self.embedder.encode([query])[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            # A minimum score lets callers say "nothing relevant found" instead of
            # being handed the least-bad chunk, which is how wrong context sneaks in.
            if min_score is not None and scores[idx] < min_score:
                continue
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