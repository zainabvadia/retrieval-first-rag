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

### Running the retrieval engine
```bash
pip install -r requirements.txt
python demo.py
```

## Part 2: AI Agent Service (`agent/`)

A backend service exposing an AI agent with real tool-calling, implementing a full decide-execute-reason loop.

- `agent/tools.py` — tool implementations (calculator, unit converter, knowledge base search) and their schema definitions for the Claude API
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

## What I'd add next

- Wire the agent's knowledge base search tool directly to the retrieval engine in Part 1, so the agent can search real documents, not just a small hardcoded dictionary
- Add re-ranking as a second retrieval pass for higher precision on ambiguous queries
- Add a rate limiter and authentication layer before any real deployment
- Add evaluation metrics (retrieval precision/recall against a labeled query set) to make retrieval quality measurable over time, not just eyeballed
