"""
evaluation.py
Measures retrieval quality against a small labeled query set, so changes to
chunking, embeddings or ranking can be compared with numbers instead of
eyeballed.

Metrics (per query, then averaged):
  hit_rate@k  - was the expected source in the top k results?
  mrr         - mean reciprocal rank of the expected source (1/rank, 0 if missing)

Run: python evaluation.py
"""

import json
import sys


def evaluate(engine, labeled_queries: list[dict], k: int = 3, mode: str = "semantic") -> dict:
    hits = 0
    reciprocal_ranks = []
    failures = []
    per_query = []

    for item in labeled_queries:
        results = engine.retrieve(item["query"], top_k=k, mode=mode)
        sources = [r["source"] for r in results]
        expected = item["expected_source"]

        if expected in sources:
            hits += 1
            rank = sources.index(expected) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            rank = None
            reciprocal_ranks.append(0.0)
            failures.append({"query": item["query"], "expected": expected, "got": sources})
        per_query.append({"query": item["query"], "expected": expected, "rank": rank, "got": sources})

    n = len(labeled_queries)
    return {
        "queries": n,
        "k": k,
        "mode": mode,
        "per_query": per_query,
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "failures": failures,
    }


def compare_modes(engine, labeled_queries: list[dict], k: int = 3) -> dict:
    """Runs the same labeled queries through every retrieval mode."""
    return {mode: evaluate(engine, labeled_queries, k=k, mode=mode) for mode in ("semantic", "keyword", "hybrid")}


def main(path: str = "eval_set.json"):
    from rag_engine import RAGEngine
    from sample_data import SAMPLE_DOCUMENTS

    with open(path) as f:
        labeled = json.load(f)

    engine = RAGEngine()
    engine.ingest(SAMPLE_DOCUMENTS)
    for mode, report in compare_modes(engine, labeled, k=3).items():
        print(f"[{mode}] queries={report['queries']} k={report['k']}  hit rate@k={report['hit_rate']:.2f}  MRR={report['mrr']:.2f}")
        for fail in report["failures"]:
            print(f"    MISS: {fail['query']!r} expected {fail['expected']}, got {fail['got']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
