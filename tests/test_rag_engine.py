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


# ---- keyword / hybrid retrieval and incremental documents ---------------------

def test_keyword_mode_matches_exact_terms_and_drops_non_matches(engine):
    hits = engine.retrieve("manager approval", top_k=5, mode="keyword")
    assert hits and all(h["keyword_score"] > 0 for h in hits)
    assert engine.retrieve("zebra quantum cheese", top_k=5, mode="keyword") == []


def test_keyword_mode_handles_plural_and_punctuation(engine):
    assert engine.retrieve("expenses?", top_k=1, mode="keyword")[0]["source"] == "expense_policy.txt"


def test_hybrid_ranks_the_right_source_first_and_reports_both_scores(engine):
    hit = engine.retrieve("How much can I spend before manager approval?", top_k=1, mode="hybrid")[0]
    assert hit["source"] == "expense_policy.txt"
    assert hit["semantic_score"] > 0 and hit["keyword_score"] > 0


def test_unknown_mode_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.retrieve("anything", mode="magic")


def test_results_carry_chunk_index(engine):
    assert all("chunk_index" in r for r in engine.retrieve("equipment", top_k=3))


def test_add_documents_makes_new_text_retrievable_without_losing_old(engine):
    before = {s["source"] for s in engine.sources()}
    engine.add_documents({"pets.txt": "Employees may bring a dog to the office on Fridays with prior notice."})
    assert engine.retrieve("bring a dog to the office", top_k=1)[0]["source"] == "pets.txt"
    assert before <= {s["source"] for s in engine.sources()}
    assert engine.retrieve("manager approval", top_k=1)[0]["source"] == "expense_policy.txt"


def test_add_documents_replaces_an_existing_source(engine):
    engine.add_documents({"expense_policy.txt": "Only zebra purchases are reimbursed."})
    assert [s for s in engine.sources() if s["source"] == "expense_policy.txt"][0]["chunks"] == 1
    assert "zebra" in engine.retrieve("zebra", top_k=1, mode="keyword")[0]["text"]


def test_remove_document(engine):
    assert engine.remove_document("expense_policy.txt") is True
    assert "expense_policy.txt" not in {s["source"] for s in engine.sources()}
    assert all(r["source"] != "expense_policy.txt" for r in engine.retrieve("manager approval", top_k=5))
    assert engine.remove_document("nope.txt") is False


def test_embeddings_stay_aligned_with_chunks_after_edits(engine):
    engine.add_documents({"a.txt": "alpha beta gamma"})
    engine.remove_document("onboarding_policy.txt")
    assert len(engine.chunks) == engine.embeddings.shape[0]
