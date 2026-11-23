import pytest
import tools


@pytest.fixture(autouse=True)
def reset_engine():
    tools.set_knowledge_engine(None)
    yield
    tools.set_knowledge_engine(None)


def test_calculator_basic_and_precedence():
    assert tools.calculator("2 + 3 * 4") == "14"
    assert tools.calculator("(2 + 3) * 4") == "20"
    assert tools.calculator("-5 + 2") == "-3"


def test_calculator_rejects_code_execution():
    assert tools.calculator("__import__('os').system('echo hi')").startswith("Error")


def test_calculator_handles_divide_by_zero():
    assert tools.calculator("1/0").startswith("Error")


def test_calculator_caps_huge_exponents():
    # Regression: 9**9**9 used to hang the process.
    assert "Exponent too large" in tools.calculator("9**9**9")
    assert tools.calculator("2**10") == "1024"


def test_unit_converter_known_and_unknown():
    assert tools.unit_converter(100, "celsius", "fahrenheit") == "100 celsius = 212.0 fahrenheit"
    assert tools.unit_converter(1, "km", "miles").startswith("1 km = 0.621")
    assert tools.unit_converter(1, "parsec", "km").startswith("Unsupported")


def test_keyword_fallback_ignores_stopwords():
    # Regression: common words like "what"/"do"/"i" used to match the wrong document.
    assert "$500" in tools.knowledge_base_search("What do I need for expenses?")


def test_keyword_fallback_reports_not_found():
    assert tools.knowledge_base_search("zebra quantum cheese").startswith("No relevant")


def test_engine_backed_search_returns_sources_and_respects_threshold():
    from rag_engine import RAGEngine, HashingEmbedder
    from sample_data import SAMPLE_DOCUMENTS
    engine = RAGEngine(embedder=HashingEmbedder())
    engine.ingest(SAMPLE_DOCUMENTS)
    tools.set_knowledge_engine(engine)

    hit = tools.knowledge_base_search("manager approval for purchases over $500")
    assert "[expense_policy.txt]" in hit
    assert tools.knowledge_base_search("zebra quantum cheese").startswith("No relevant")
