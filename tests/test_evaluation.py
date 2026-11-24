import json
import os
import pytest
from evaluation import evaluate
from rag_engine import RAGEngine, HashingEmbedder
from sample_data import SAMPLE_DOCUMENTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def engine_and_labels():
    engine = RAGEngine(embedder=HashingEmbedder())
    engine.ingest(SAMPLE_DOCUMENTS)
    with open(os.path.join(ROOT, "eval_set.json")) as f:
        return engine, json.load(f)


class StubEngine:
    def __init__(self, sources):
        self.sources = sources

    def retrieve(self, query, top_k=3, mode="semantic"):
        return [{"source": s, "text": "", "score": 1.0} for s in self.sources[:top_k]]


def test_hit_rate_and_mrr_with_known_ranks():
    labeled = [
        {"query": "q1", "expected_source": "a"},   # rank 1 -> rr 1.0
        {"query": "q2", "expected_source": "b"},   # rank 2 -> rr 0.5
        {"query": "q3", "expected_source": "zzz"}, # missing -> 0
    ]
    report = evaluate(StubEngine(["a", "b", "c"]), labeled, k=3)
    # Same stub for every query: q1 finds "a" at rank 1, q2 finds "b" at rank 2, q3 misses.
    assert report["hit_rate"] == 2 / 3
    assert abs(report["mrr"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9
    assert report["failures"][0]["query"] == "q3"


def test_empty_query_set_is_safe():
    assert evaluate(StubEngine(["a"]), [], k=3)["hit_rate"] == 0.0


def test_labeled_set_meets_a_quality_floor():
    engine = RAGEngine(embedder=HashingEmbedder())
    engine.ingest(SAMPLE_DOCUMENTS)
    with open(os.path.join(ROOT, "eval_set.json")) as f:
        labeled = json.load(f)
    report = evaluate(engine, labeled, k=3)
    assert report["hit_rate"] >= 0.9  # fails if a change makes retrieval worse


def test_evaluate_reports_per_query_ranks_and_mode(engine_and_labels):
    engine, labeled = engine_and_labels
    report = evaluate(engine, labeled, k=3, mode="hybrid")
    assert report["mode"] == "hybrid"
    assert len(report["per_query"]) == report["queries"]
    assert all(q["rank"] is None or 1 <= q["rank"] <= 3 for q in report["per_query"])


def test_compare_modes_covers_every_mode(engine_and_labels):
    from evaluation import compare_modes
    engine, labeled = engine_and_labels
    out = compare_modes(engine, labeled, k=3)
    assert set(out) == {"semantic", "keyword", "hybrid"}
    assert all(0.0 <= r["mrr"] <= r["hit_rate"] <= 1.0 for r in out.values())
