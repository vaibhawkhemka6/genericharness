"""FastAPI app exposing the Agent as a streaming (SSE) chat API."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app import config
from app.agent.agent import Agent
from app.models.anthropic import Claude
from app.models.base import Model
from app.models.openai import OpenAIModel
from app.session.memory import InMemorySessionStore
from app.tools.calculator import calculator
from app.tools.datetime_tool import get_current_datetime
from app.tools.web_search import web_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are the HealthHarness assistant, an early-stage general-purpose agent. "
    "Use tools when they help you answer accurately (search the web for facts you "
    "don't know, use the calculator for arithmetic). Be concise and clear."
)


def build_model() -> Model:
    if config.DEFAULT_MODEL_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("DEFAULT_MODEL_PROVIDER=openai but OPENAI_API_KEY is not set.")
        return OpenAIModel(id=config.DEFAULT_MODEL_ID)
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("DEFAULT_MODEL_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
    return Claude(id=config.DEFAULT_MODEL_ID)


def build_agent() -> Agent:
    return Agent(
        model=build_model(),
        tools=[web_search, calculator, get_current_datetime],
        instructions=INSTRUCTIONS,
    )


app = FastAPI(title="HealthHarness Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store = InMemorySessionStore()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> EventSourceResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        agent = build_agent()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    history = session_store.get_history(req.session_id)

    def event_generator():
        for event in agent.run(history, req.message):
            yield {"event": event.type, "data": json.dumps(event.to_dict())}

    return EventSourceResponse(event_generator())


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, str]:
    session_store.reset(session_id)
    return {"status": "reset"}
