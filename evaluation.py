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


def evaluate(engine, labeled_queries: list[dict], k: int = 3) -> dict:
    hits = 0
    reciprocal_ranks = []
    failures = []

    for item in labeled_queries:
        results = engine.retrieve(item["query"], top_k=k)
        sources = [r["source"] for r in results]
        expected = item["expected_source"]

        if expected in sources:
            hits += 1
            reciprocal_ranks.append(1.0 / (sources.index(expected) + 1))
        else:
            reciprocal_ranks.append(0.0)
            failures.append({"query": item["query"], "expected": expected, "got": sources})

    n = len(labeled_queries)
    return {
        "queries": n,
        "k": k,
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "failures": failures,
    }


def main(path: str = "eval_set.json"):
    from rag_engine import RAGEngine
    from sample_data import SAMPLE_DOCUMENTS

    with open(path) as f:
        labeled = json.load(f)

    engine = RAGEngine()
    engine.ingest(SAMPLE_DOCUMENTS)
    report = evaluate(engine, labeled, k=3)

    print(f"Queries: {report['queries']}  k={report['k']}")
    print(f"Hit rate@{report['k']}: {report['hit_rate']:.2f}")
    print(f"MRR: {report['mrr']:.2f}")
    for fail in report["failures"]:
        print(f"MISS: {fail['query']!r} expected {fail['expected']}, got {fail['got']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
