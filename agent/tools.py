"""
tools.py
Tool definitions the agent can call. Each tool is a real, working function,
not a mock, the agent decides which one(s) it needs based on the user's request,
calls it, and uses the real result to form its answer.

Design principle: tools should be narrow and single-purpose. A tool that tries
to do too much is harder for the model to reason about reliably, and harder
to test and debug when something goes wrong.
"""

import ast
import operator


# ---- Tool 1: Calculator ----
# Safe expression evaluation, no eval() on raw strings, since that's a real
# security risk if this were ever exposed to untrusted input.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


_MAX_EXPONENT = 100


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        # Without a cap, an input like 9**9**9 would hang the service (denial of service).
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {_MAX_EXPONENT})")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {node}")


def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression safely (no eval() on raw input)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ---- Tool 2: Unit converter ----
_CONVERSIONS = {
    ("km", "miles"): lambda x: x * 0.621371,
    ("miles", "km"): lambda x: x / 0.621371,
    ("kg", "lbs"): lambda x: x * 2.20462,
    ("lbs", "kg"): lambda x: x / 2.20462,
    ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
}


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Converts a value between common units (km/miles, kg/lbs, celsius/fahrenheit)."""
    key = (from_unit.lower(), to_unit.lower())
    if key not in _CONVERSIONS:
        return f"Unsupported conversion: {from_unit} to {to_unit}"
    result = _CONVERSIONS[key](value)
    return f"{value} {from_unit} = {round(result, 3)} {to_unit}"


# ---- Tool 3: Knowledge base search ----
# A small in-memory document store, standing in for the kind of retrieval
# system built in the companion RAG project. Kept simple and dependency-light
# here since this project's focus is the agent/tool-calling layer, not retrieval
# depth, that's already demonstrated in the retrieval-first-rag project.
_KNOWLEDGE_BASE = {
    "onboarding": "New employees complete onboarding within their first two weeks, including workstation setup and security training.",
    "expenses": "Expense reports up to $500 don't require manager approval. Anything above that needs written approval before purchase.",
    "remote_work": "Remote employees are expected to be available during core hours of 10am-3pm local time.",
}


_STOPWORDS = {"the", "a", "an", "is", "are", "do", "does", "to", "be", "of", "in", "on", "for",
              "and", "or", "what", "when", "where", "how", "need", "needs", "i", "my"}


# Optional retrieval engine (see rag_engine.py). When set, the knowledge base tool
# searches real documents with embeddings instead of the keyword fallback below.
_engine = None
MIN_SCORE = 0.3  # below this, say "not found" rather than returning weak context


def set_knowledge_engine(engine) -> None:
    global _engine
    _engine = engine


def knowledge_base_search(query: str) -> str:
    """
    Searches the knowledge base. Uses the retrieval engine when one is configured
    (returning the source for each passage so answers can be traced), and otherwise
    falls back to simple keyword overlap that ignores common stopwords.
    """
    if _engine is not None:
        results = _engine.retrieve(query, top_k=2, min_score=MIN_SCORE)
        if not results:
            return "No relevant information found in the knowledge base."
        return "\n".join(f"[{r['source']}] {r['text'].strip()}" for r in results)
    query_words = {w for w in query.lower().split() if w not in _STOPWORDS}
    best_match = None
    best_score = 0
    for topic, content in _KNOWLEDGE_BASE.items():
        content_words = set(content.lower().replace(".", "").replace(",", "").replace("$", "").split())
        topic_words = set(topic.replace("_", " ").split())
        score = len(query_words & (content_words | topic_words))
        if score > best_score:
            best_score = score
            best_match = content
    return best_match if best_match else "No relevant information found in the knowledge base."


# Tool schema definitions for the Claude API (tool-calling format)
TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "Evaluates a basic arithmetic expression. Use for any math the user asks for.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "A math expression, e.g. '3 * (4 + 2)'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "unit_converter",
        "description": "Converts a value between units (km/miles, kg/lbs, celsius/fahrenheit).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "knowledge_base_search",
        "description": "Searches an internal company knowledge base for policy information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

TOOL_FUNCTIONS = {
    "calculator": lambda args: calculator(args["expression"]),
    "unit_converter": lambda args: unit_converter(args["value"], args["from_unit"], args["to_unit"]),
    "knowledge_base_search": lambda args: knowledge_base_search(args["query"]),
}
