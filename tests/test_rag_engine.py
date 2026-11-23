import pytest
from rag_engine import RAGEngine, HashingEmbedder
from sample_data import SAMPLE_DOCUMENTS


@pytest.fixture
def engine():
    e = RAGEngine(embedder=HashingEmbedder())
    e.ingest(SAMPLE_DOCUMENTS)
    return e


def test_chunks_overlap_so_boundary_facts_survive(engine):
    words = [f"w{i}" for i in range(500)]
    chunks = engine.chunk_text(" ".join(words), chunk_size=200, overlap=40)
    assert len(chunks) >= 3
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-40:] == second[:40]


def test_chunking_covers_the_whole_text(engine):
    words = [f"w{i}" for i in range(450)]
    chunks = engine.chunk_text(" ".join(words), chunk_size=200, overlap=40)
    assert "w0" in chunks[0].split()
    assert "w449" in chunks[-1].split()


def test_short_text_is_a_single_chunk(engine):
    assert len(engine.chunk_text("just a few words")) == 1


def test_retrieval_ranks_the_right_source_first(engine):
    top = engine.retrieve("How much can I spend before manager approval?", top_k=1)[0]
    assert top["source"] == "expense_policy.txt"


def test_results_are_sorted_by_score(engine):
    scores = [r["score"] for r in engine.retrieve("remote equipment shipped", top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_min_score_filters_out_irrelevant_queries(engine):
    assert engine.retrieve("zebra quantum cheese", top_k=3, min_score=0.2) == []


def test_retrieve_before_ingest_raises():
    with pytest.raises(ValueError):
        RAGEngine(embedder=HashingEmbedder()).retrieve("anything")


def test_index_save_and_load_roundtrip(engine, tmp_path):
    path = str(tmp_path / "index.json")
    engine.save_index(path)
    fresh = RAGEngine(embedder=HashingEmbedder())
    fresh.load_index(path)
    q = "How long does reimbursement take?"
    assert fresh.retrieve(q, top_k=1)[0]["source"] == engine.retrieve(q, top_k=1)[0]["source"]
