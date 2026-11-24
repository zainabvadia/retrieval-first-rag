import pytest
from fastapi.testclient import TestClient

import main as service
from rag_engine import RAGEngine, HashingEmbedder
from sample_data import SAMPLE_DOCUMENTS


@pytest.fixture
def client(monkeypatch):
    engine = RAGEngine(embedder=HashingEmbedder())
    engine.ingest(SAMPLE_DOCUMENTS)
    monkeypatch.setattr(service, "_engine", engine)
    return TestClient(service.app)


def test_status_reports_embedder_and_counts(client):
    body = client.get("/api/status").json()
    assert body["documents"] == len(SAMPLE_DOCUMENTS)
    assert body["chunks"] >= body["documents"]
    assert "embedder" in body


@pytest.mark.parametrize("mode", ["semantic", "keyword", "hybrid"])
def test_search_returns_scored_results_in_every_mode(client, mode):
    r = client.post("/api/search", json={"query": "manager approval for purchases", "top_k": 2, "mode": mode})
    assert r.status_code == 200
    results = r.json()["results"]
    assert 1 <= len(results) <= 2
    assert results[0]["source"] == "expense_policy.txt"
    assert {"text", "score", "chunk_index", "semantic_score", "keyword_score"} <= set(results[0])


def test_search_validates_input(client):
    assert client.post("/api/search", json={"query": "  "}).status_code == 400
    assert client.post("/api/search", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/api/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_add_search_and_delete_a_document(client):
    r = client.post("/api/documents", json={"source": "pets.txt", "text": "Dogs are welcome in the office on Fridays."})
    assert r.status_code == 200
    assert "pets.txt" in [d["source"] for d in r.json()["documents"]]

    top = client.post("/api/search", json={"query": "dogs in the office", "top_k": 1}).json()["results"][0]
    assert top["source"] == "pets.txt"

    assert client.delete("/api/documents/pets.txt").status_code == 200
    assert client.delete("/api/documents/pets.txt").status_code == 404


def test_empty_document_is_rejected(client):
    assert client.post("/api/documents", json={"source": "a.txt", "text": "   "}).status_code == 400


def test_searching_with_no_documents_gives_a_clear_error(client):
    for d in list(client.get("/api/documents").json()["documents"]):
        client.delete(f"/api/documents/{d['source']}")
    assert client.post("/api/search", json={"query": "anything"}).status_code == 400


def test_evaluate_compares_all_modes(client):
    body = client.post("/api/evaluate", json={"k": 3}).json()
    assert set(body) == {"semantic", "keyword", "hybrid"}
    assert 0.0 <= body["hybrid"]["hit_rate"] <= 1.0


def test_evaluate_skips_queries_whose_document_was_removed(client):
    client.delete("/api/documents/expense_policy.txt")
    body = client.post("/api/evaluate", json={"k": 3}).json()
    assert body["semantic"]["queries"] < 8


def test_ui_page_is_served(client):
    r = client.get("/ui")
    assert r.status_code == 200
    assert "html" in r.text.lower()
