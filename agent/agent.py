"""
agent.py
Core agent logic: given a user message, the agent decides whether it needs
a tool to answer accurately, calls the tool if so, and incorporates the real
result into its final response. This is the actual "agentic" pattern, not
just a single LLM call, a loop where the model can act, observe a real
result, and reason further before responding.
"""

import os
import anthropic
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

MODEL = "claude-sonnet-4-5"


class Agent:
    def __init__(self, client=None):
        # The client can be injected so the loop can be tested without network calls.
        self.client = client if client is not None else anthropic.Anthropic()

    def run(self, user_message: str, max_turns: int = 4) -> dict:
        """
        Runs the agent loop: send the message, check if Claude wants to call
        a tool, execute it for real, feed the result back, repeat until
        Claude gives a final text answer or max_turns is hit.

        Returns a dict with the final answer and a trace of every tool call
        made along the way, useful for debugging and for demoing exactly
        what the agent did, not just what it said.
        """
        messages = [{"role": "user", "content": user_message}]
        trace = []

        for _ in range(max_turns):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # If Claude didn't ask for a tool, we have our final answer
            if response.stop_reason != "tool_use":
                final_text = "".join(block.text for block in response.content if block.type == "text")
                return {"answer": final_text, "tool_calls": trace}

            # Claude wants to use one or more tools, execute them for real
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    # A bad tool name or a crashing tool must not take down the whole loop:
                    # report the error back to the model so it can recover or explain.
                    try:
                        result = TOOL_FUNCTIONS[tool_name](tool_input)
                    except Exception as e:
                        result = f"Tool error ({tool_name}): {e}"

                    trace.append({"tool": tool_name, "input": tool_input, "result": result})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            messages.append({"role": "user", "content": tool_results})

        return {"answer": "Reached max turns without a final answer.", "tool_calls": trace}
