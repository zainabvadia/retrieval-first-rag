"""
demo.py
Command-line demo of the RAG engine using a small set of sample documents.

Run: python demo.py

If ANTHROPIC_API_KEY is set in your environment, this will also generate
a natural-language answer grounded in the retrieved context using Claude.
Without a key, it runs in retrieval-only mode and shows exactly what
context would be handed to an LLM, which is often the more instructive
view when evaluating a RAG system's actual quality.
"""

import os
from rag_engine import RAGEngine
from sample_data import SAMPLE_DOCUMENTS

def generate_answer_with_claude(query: str, context_chunks: list[dict]) -> str:
    """
    Optional generation step using the Anthropic API.
    Only runs if ANTHROPIC_API_KEY is set, this keeps the demo runnable
    by anyone without requiring a paid API key just to see retrieval work.
    """
    import anthropic

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Answer the question using ONLY the context below. "
                f"If the context doesn't contain the answer, say so.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}"
            ),
        }],
    )
    return message.content[0].text


def main():
    print("Building index from sample documents...\n")
    engine = RAGEngine()
    engine.ingest(SAMPLE_DOCUMENTS)

    queries = [
        "How much can I spend before needing manager approval?",
        "When do remote employees get their equipment?",
        "What hours do I need to be online if I work remotely?",
    ]

    for query in queries:
        print(f"Q: {query}")
        results = engine.retrieve(query, top_k=2)

        print("Retrieved context:")
        for r in results:
            print(f"  [{r['source']}, score={r['score']:.3f}] {r['text'].strip()[:150]}...")

        if os.environ.get("ANTHROPIC_API_KEY"):
            answer = generate_answer_with_claude(query, results)
            print(f"\nGenerated answer: {answer}")
        else:
            print("\n(Set ANTHROPIC_API_KEY to see a generated natural-language answer here.)")

        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()