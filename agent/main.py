"""
main.py
FastAPI backend service exposing the agent as a real HTTP API.
This is the "software engineering" half of the project, a properly
structured service with typed request/response models, health checks,
and error handling, not just a script.

Run: uvicorn main:app --reload
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
import sys
from agent import Agent
import tools

app = FastAPI(title="Agent Service", description="A backend service exposing an AI agent with real tool-calling capabilities.")

# Allow the standalone frontend (index.html, served separately or opened directly)
# to call this API. In a real production deployment, this would be locked down
# to a specific known origin, not left wide open.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_embedder_name = "none"


def _load_knowledge_engine():
    """
    Builds the retrieval engine over the sample documents.

    Prefers the real sentence-embedding model. If it is not installed, falls back
    to the dependency-free HashingEmbedder (word overlap only, no meaning) so the
    API and UI still work, and reports which embedder is active via /api/status
    so a weaker fallback is never mistaken for the real thing.
    """
    global _embedder_name
    from rag_engine import RAGEngine, HashingEmbedder
    from sample_data import SAMPLE_DOCUMENTS
    try:
        engine = RAGEngine()
        engine.ingest(SAMPLE_DOCUMENTS)
        _embedder_name = "all-MiniLM-L6-v2 (sentence-transformers)"
        return engine
    except Exception as e:
        print(f"Embedding model unavailable ({e}); using the word-overlap fallback embedder.")
        engine = RAGEngine(embedder=HashingEmbedder())
        engine.ingest(SAMPLE_DOCUMENTS)
        _embedder_name = "hashing fallback (word overlap only)"
        return engine


_engine = _load_knowledge_engine()
tools.set_knowledge_engine(_engine)

agent = Agent()


class AgentRequest(BaseModel):
    message: str


class ToolCallRecord(BaseModel):
    tool: str
    input: dict
    result: str


class AgentResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord]


@app.get("/health")
def health_check():
    """Basic health check endpoint, standard practice for any deployed service."""
    return {"status": "ok"}


@app.post("/agent", response_model=AgentResponse)
def run_agent(request: AgentRequest):
    """
    Main agent endpoint. Takes a user message, runs it through the agent's
    tool-calling loop, and returns both the final answer and a full trace
    of which tools were called and what they returned, useful for
    transparency and debugging, not just a black-box answer.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    result = agent.run(request.message)
    return AgentResponse(answer=result["answer"], tool_calls=result["tool_calls"])


# ---- Retrieval API (powers the /ui page) --------------------------------------

Mode = Literal["semantic", "keyword", "hybrid"]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(3, ge=1, le=20)
    mode: Mode = "semantic"
    min_score: float | None = None


class DocumentRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=120)
    text: str


class EvaluateRequest(BaseModel):
    k: int = Field(3, ge=1, le=10)


def _require_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Retrieval engine is not available.")
    return _engine


@app.get("/api/status")
def status():
    engine = _require_engine()
    return {"embedder": _embedder_name, "documents": len(engine.sources()), "chunks": len(engine.chunks)}


@app.get("/api/documents")
def list_documents():
    return {"documents": _require_engine().sources()}


@app.post("/api/documents")
def add_document(request: DocumentRequest):
    """Adds (or replaces) a document and indexes it immediately."""
    engine = _require_engine()
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")
    engine.add_documents({request.source.strip(): request.text})
    return {"documents": engine.sources()}


@app.delete("/api/documents/{source}")
def delete_document(source: str):
    engine = _require_engine()
    if not engine.remove_document(source):
        raise HTTPException(status_code=404, detail="No such document.")
    return {"documents": engine.sources()}


@app.post("/api/search")
def search(request: SearchRequest):
    engine = _require_engine()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not engine.chunks:
        raise HTTPException(status_code=400, detail="No documents are indexed. Add one first.")
    return {"mode": request.mode,
            "results": engine.retrieve(request.query, top_k=request.top_k, min_score=request.min_score, mode=request.mode)}


@app.post("/api/evaluate")
def run_evaluation(request: EvaluateRequest):
    """Scores every retrieval mode on the labeled query set (eval_set.json)."""
    import json
    from evaluation import compare_modes
    engine = _require_engine()
    with open(os.path.join(ROOT, "eval_set.json")) as f:
        labeled = json.load(f)
    # Only score queries whose expected document is still indexed, so deleting a
    # document in the UI does not produce misleading "misses".
    present = {s["source"] for s in engine.sources()}
    labeled = [q for q in labeled if q["expected_source"] in present]
    if not labeled:
        raise HTTPException(status_code=400, detail="None of the labeled queries match the indexed documents.")
    return compare_modes(engine, labeled, k=request.k)


@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(os.path.join(ROOT, "ui", "index.html"))


@app.get("/")
def root():
    return {
        "service": "Agent Service",
        "endpoints": {
            "GET /ui": "Retrieval explorer (search, documents, evaluation)",
            "POST /agent": "Send a message to the agent",
            "POST /api/search": "Search the knowledge base (semantic, keyword or hybrid)",
            "GET /health": "Health check",
        },
    }
