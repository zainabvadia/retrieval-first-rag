from fastapi.testclient import TestClient
import main as service


class StubAgent:
    def run(self, message):
        return {"answer": f"echo: {message}", "tool_calls": [{"tool": "calculator", "input": {"expression": "1+1"}, "result": "2"}]}


def make_client(monkeypatch):
    monkeypatch.setattr(service, "agent", StubAgent())
    return TestClient(service.app)


def test_health(monkeypatch):
    assert make_client(monkeypatch).get("/health").json() == {"status": "ok"}


def test_agent_returns_answer_and_trace(monkeypatch):
    r = make_client(monkeypatch).post("/agent", json={"message": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo: hi"
    assert body["tool_calls"][0]["tool"] == "calculator"


def test_empty_message_is_rejected(monkeypatch):
    assert make_client(monkeypatch).post("/agent", json={"message": "   "}).status_code == 400


def test_missing_message_is_a_validation_error(monkeypatch):
    assert make_client(monkeypatch).post("/agent", json={}).status_code == 422
