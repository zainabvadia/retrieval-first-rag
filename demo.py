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

SAMPLE_DOCUMENTS = {
    "onboarding_policy.txt": """
        All new employees must complete their onboarding checklist within
        the first two weeks of their start date. This includes setting up
        their workstation, completing mandatory security training, and
        meeting with their assigned manager for a 1:1 introduction.
        Remote employees receive equipment shipped to their home address
        within 5 business days of their offer acceptance. IT support is
        available Monday through Friday, 9am to 6pm EST, for any setup
        issues during onboarding.
    """,
    "expense_policy.txt": """
        Employees may submit expense reports for business-related travel,
        meals, and equipment purchases up to $500 without prior manager
        approval. Purchases exceeding $500 require written approval from
        a direct manager before the purchase is made. All expense reports
        must be submitted within 30 days of the purchase date, along with
        an itemized receipt. Reimbursement is typically processed within
        10 business days of report submission.
    """,
    "remote_work_policy.txt": """
        Employees working remotely are expected to be available during
        core collaboration hours of 10am to 3pm in their local timezone.
        Remote employees must have a stable internet connection capable
        of supporting video calls, and are encouraged to work from a
        quiet, dedicated workspace. Requests for fully remote arrangements
        must be approved by both the employee's manager and HR.
    """,
}


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