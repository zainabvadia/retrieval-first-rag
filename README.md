# Retrieval-First RAG & AI Agent

A retrieval-augmented generation engine, plus an AI agent service with real tool-calling built on top of it. Built to demonstrate that retrieval quality, not the language model, is usually the actual bottleneck in RAG systems, and that agentic AI needs a real decide-execute-reason loop, not just a single prompt-response call.

## Why I built it this way

Most RAG failures I've encountered aren't caused by the language model being "not smart enough," they're caused by the retrieval step handing the model the wrong context in the first place. This project is deliberately structured to separate retrieval from generation, so retrieval quality is inspectable on its own, a lesson I learned building a real RAG system for pharmaceutical compliance document search, where retrieval accuracy directly determined whether the final answer could be trusted.

The agent service extends this further: an AI agent that can use tools, a calculator, a unit converter, and a knowledge base search (which can be swapped for the RAG engine above), deciding when a tool is actually needed, executing it for real, and reasoning over the real result before answering. That's the difference between a chatbot wrapper and an actual agentic system.

## Part 1: Retrieval Engine (`rag_engine.py`, `demo.py`)

1. **Chunking** — documents are split into overlapping word-based chunks. Overlap matters: without it, a fact spanning a chunk boundary can become unretrievable.
2. **Embedding** — each chunk is embedded using `all-MiniLM-L6-v2`, a small, fast, open-source sentence embedding model.
3. **Retrieval** — a query is embedded the same way, and the most semantically similar chunks are returned via cosine similarity, ranked by score.
4. **Generation (optional)** — if `ANTHROPIC_API_KEY` is set, retrieved chunks are passed to Claude to generate a grounded answer. Without a key, the demo still runs and shows exactly what context would be handed to the model.

The embedder is pluggable (`SentenceTransformerEmbedder` by default). A tiny `HashingEmbedder` lets the tests run offline in under two seconds; it only matches word overlap, so it checks the engine's logic (chunking, ranking, thresholds), not embedding quality. `retrieve()` accepts a `min_score` so the system can say "nothing relevant found" instead of passing weak context to the model.

### Measuring retrieval quality
`evaluation.py` scores retrieval against a labeled query set (`eval_set.json`) using hit rate@k and mean reciprocal rank (MRR), so a change to chunk size or the embedding model can be compared with numbers.
```bash
python evaluation.py
```

### Running the retrieval engine
```bash
pip install -r requirements.txt
python demo.py
```

## Part 2: AI Agent Service (`agent/`)

A backend service exposing an AI agent with real tool-calling, implementing a full decide-execute-reason loop.

- `agent/tools.py` — tool implementations (calculator, unit converter, knowledge base search) and their schema definitions for the Claude API. The knowledge base tool searches the Part 1 retrieval engine when it is available (returning the source of each passage, with a score threshold) and falls back to keyword search otherwise
- `agent/agent.py` — the core tool-calling loop
- `agent/main.py` — FastAPI service exposing the agent as a real HTTP API, with CORS enabled
- `agent/index.html` — a standalone web UI showing the agent's answer alongside a live trace of every tool call and result

### Running the agent service
```bash
cd agent
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload
```
Then open `agent/index.html` directly in your browser, it connects to the running API automatically.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
27 tests run on every push through GitHub Actions. They cover chunk overlap, ranking and score thresholds, index save/load, the evaluation metrics, calculator safety (code injection, divide by zero, huge exponents), the agent loop using a scripted fake model client, and the HTTP API. Writing them found two real bugs: `9**9**9` hung the calculator, and the keyword fallback missed words with punctuation (`"expenses?"`).

**Limits:** the tests use the hashing embedder and a fake model client, so they verify the logic, not real embedding quality or live model behavior. The evaluation set is small (8 queries over 3 documents).

## What I'd add next

- Add re-ranking as a second retrieval pass for higher precision on ambiguous queries
- Add a rate limiter and authentication layer before any real deployment
- Grow the evaluation set and track hit rate and MRR across embedding models and chunk sizes
