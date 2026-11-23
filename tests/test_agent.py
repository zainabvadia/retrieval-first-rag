from types import SimpleNamespace as NS
from agent import Agent


def text_response(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


def tool_response(name, args, call_id="t1"):
    return NS(stop_reason="tool_use", content=[NS(type="tool_use", name=name, input=args, id=call_id)])


class FakeClient:
    """Plays back scripted model responses so the loop can be tested offline."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_answers_directly_when_no_tool_is_needed():
    out = Agent(client=FakeClient([text_response("Hello!")])).run("hi")
    assert out == {"answer": "Hello!", "tool_calls": []}


def test_runs_a_tool_then_answers_with_the_real_result():
    client = FakeClient([
        tool_response("calculator", {"expression": "6 * 7"}),
        text_response("The answer is 42."),
    ])
    out = Agent(client=client).run("What is 6 times 7?")
    assert out["answer"] == "The answer is 42."
    assert out["tool_calls"] == [{"tool": "calculator", "input": {"expression": "6 * 7"}, "result": "42"}]
    # The tool result must be fed back to the model on the second call.
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and sent["content"] == "42"


def test_unknown_tool_is_reported_back_instead_of_crashing():
    client = FakeClient([tool_response("launch_rocket", {}), text_response("I can't do that.")])
    out = Agent(client=client).run("go")
    assert out["tool_calls"][0]["result"].startswith("Tool error")
    assert out["answer"] == "I can't do that."


def test_stops_after_max_turns():
    client = FakeClient([tool_response("calculator", {"expression": "1+1"}, f"t{i}") for i in range(10)])
    out = Agent(client=client).run("loop forever", max_turns=3)
    assert out["answer"].startswith("Reached max turns")
    assert len(out["tool_calls"]) == 3
