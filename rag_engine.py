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
        self._keyword_index = None  # built lazily, reset whenever chunks change

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
        Replaces anything ingested before; use add_documents() to extend.
        """
        self.chunks = []
        for source, text in documents.items():
            for i, chunk in enumerate(self.chunk_text(text)):
                self.chunks.append({"source": source, "text": chunk, "chunk_index": i})

        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.embedder.encode(texts) if texts else None
        self._keyword_index = None

    def add_documents(self, documents: dict[str, str]):
        """
        Adds or replaces documents without re-embedding the ones already
        indexed. A source name that already exists is replaced.
        """
        if not documents:
            return
        for source in documents:
            self._drop_source(source)
        new_chunks = []
        for source, text in documents.items():
            for i, chunk in enumerate(self.chunk_text(text)):
                new_chunks.append({"source": source, "text": chunk, "chunk_index": i})
        if not new_chunks:
            return
        new_emb = self.embedder.encode([c["text"] for c in new_chunks])
        self.embeddings = new_emb if self.embeddings is None or len(self.chunks) == 0 else np.vstack([self.embeddings, new_emb])
        self.chunks.extend(new_chunks)
        self._keyword_index = None

    def remove_document(self, source: str) -> bool:
        """Removes a document and its chunks. Returns False if it was not indexed."""
        removed = self._drop_source(source)
        if removed:
            self._keyword_index = None
        return removed

    def _drop_source(self, source: str) -> bool:
        keep = [i for i, c in enumerate(self.chunks) if c["source"] != source]
        if len(keep) == len(self.chunks):
            return False
        self.chunks = [self.chunks[i] for i in keep]
        self.embeddings = self.embeddings[keep] if self.embeddings is not None and keep else None
        return True

    def sources(self) -> list[dict]:
        """One entry per indexed document: name, chunk count and word count."""
        out: dict[str, dict] = {}
        for c in self.chunks:
            s = out.setdefault(c["source"], {"source": c["source"], "chunks": 0, "words": 0})
            s["chunks"] += 1
            s["words"] += len(c["text"].split())
        return list(out.values())

    # ---- keyword (BM25) scoring -------------------------------------------------

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9$]+", text.lower())
        words = [w for w in words if w not in HashingEmbedder._STOP]
        # light plural stemming so "expenses" matches "expense"
        return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]

    def _build_keyword_index(self):
        docs = [self._tokens(c["text"]) for c in self.chunks]
        df: dict[str, int] = {}
        for toks in docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        avg = sum(len(d) for d in docs) / len(docs) if docs else 0.0
        self._keyword_index = {"docs": docs, "df": df, "avg": avg}

    def _keyword_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """BM25 over the chunks. Rewards rare query words, saturates repeats, normalises for length."""
        if self._keyword_index is None:
            self._build_keyword_index()
        idx = self._keyword_index
        n = len(idx["docs"])
        q_terms = set(self._tokens(query))
        scores = np.zeros(n, dtype=np.float64)
        for i, toks in enumerate(idx["docs"]):
            if not toks:
                continue
            for term in q_terms:
                f = toks.count(term)
                if not f:
                    continue
                df = idx["df"].get(term, 0)
                idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
                scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(toks) / (idx["avg"] or 1.0)))
        return scores

    # ---- retrieval -------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3, min_score: float | None = None, mode: str = "semantic") -> list[dict]:
        """
        Returns the top_k most relevant chunks to the query, ranked best first.

        mode="semantic" (default): cosine similarity between embeddings. Catches
            paraphrases ("who do I tell?" vs "report to the security team") but
            can miss exact terms such as codes, names and numbers.
        mode="keyword": BM25 word matching. The reverse: great on exact terms,
            blind to meaning. Chunks that share no word with the query are dropped.
        mode="hybrid": reciprocal rank fusion of the two rankings, so a chunk that
            both methods like rises to the top.

        min_score applies to the semantic similarity only, so the system can say
        "nothing relevant found" instead of passing weak context to the model.
        Every result also carries chunk_index and the per-method scores.
        """
        if mode not in ("semantic", "keyword", "hybrid"):
            raise ValueError(f"Unknown retrieval mode: {mode!r}")
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("No documents ingested yet. Call ingest() first.")

        query_embedding = self.embedder.encode([query])[0]
        semantic = self.embeddings @ query_embedding
        keyword = self._keyword_scores(query)

        if mode == "semantic":
            order = list(np.argsort(semantic)[::-1])
            combined = semantic
        elif mode == "keyword":
            order = [i for i in np.argsort(keyword)[::-1] if keyword[i] > 0]
            combined = keyword
        else:
            k_rank = {int(i): r for r, i in enumerate(np.argsort(keyword)[::-1]) if keyword[i] > 0}
            s_rank = {int(i): r for r, i in enumerate(np.argsort(semantic)[::-1])}
            rrf = np.zeros(len(self.chunks))
            for i in range(len(self.chunks)):
                rrf[i] = 1.0 / (60 + s_rank[i] + 1) + (1.0 / (60 + k_rank[i] + 1) if i in k_rank else 0.0)
            order = list(np.argsort(rrf)[::-1])
            combined = rrf

        results = []
        for idx in order:
            if len(results) >= top_k:
                break
            # A minimum score lets callers say "nothing relevant found" instead of
            # being handed the least-bad chunk, which is how wrong context sneaks in.
            if min_score is not None and semantic[idx] < min_score:
                continue
            results.append({
                "source": self.chunks[idx]["source"],
                "text": self.chunks[idx]["text"],
                "chunk_index": self.chunks[idx].get("chunk_index", 0),
                "score": float(combined[idx]),
                "semantic_score": float(semantic[idx]),
                "keyword_score": float(keyword[idx]),
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
        self._keyword_index = None
        self.embeddings = np.array(data["embeddings"])