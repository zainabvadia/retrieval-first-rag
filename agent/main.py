"""
main.py
FastAPI backend service exposing the agent as a real HTTP API.
This is the "software engineering" half of the project, a properly
structured service with typed request/response models, health checks,
and error handling, not just a script.

Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import Agent

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


@app.get("/")
def root():
    return {
        "service": "Agent Service",
        "endpoints": {"POST /agent": "Send a message to the agent", "GET /health": "Health check"},
    }
