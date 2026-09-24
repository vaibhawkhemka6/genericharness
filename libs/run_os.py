"""The thing you actually run: `python run_os.py` boots a real OS process -
two demo agents, a real (file-backed, not `:memory:`) `SqliteDb` shared
between them, served over HTTP by `create_app()`/`serve()` (`wind/os/app.py`).

Not a `stageN_*.py` smoke-test script - this is the equivalent of what a
real deployment's entrypoint looks like, kept at the repo root next to the
stage scripts because it's the natural "how do I actually run this" answer
once the OS layer exists, the same role `app/main.py` played for the
original Phase 1 prototype before the Stage 0 rebuild. `stage5_os.py` is
where the actual verification lives; this file is for a human hitting it
with curl.

Run: python run_os.py
Then, in another terminal:
  curl http://127.0.0.1:7777/health
  curl -X POST http://127.0.0.1:7777/agents/weather_agent/runs \\
    -H "Content-Type: application/json" \\
    -d '{"message": "What is the weather in Paris?", "session_id": "demo-1"}'
"""

from dotenv import load_dotenv

load_dotenv(override=True)

from wind.agent.agent import Agent
from wind.db.sqlite.sqlite import SqliteDb
from wind.models.anthropic.claude import Claude
from wind.os.app import create_app, serve
from wind.tools.decorator import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. Paris.
    """
    return f"22 celsius and sunny in {city}"


def build_agents(db: SqliteDb) -> list:
    """Every agent gets an explicit, stable `id=` (see `os/registry.py`'s
    docstring for why that's required, not just convention) and the same
    shared `db` - so both `Agent.run()`'s own persistence and
    `GET /sessions`'s listing (which reads `create_app()`'s separate `db=`
    parameter, not any one agent's) see the same rows.
    """
    return [
        Agent(
            id="weather_agent",
            model=Claude(),
            description="You are a helpful weather assistant. Use tools when relevant.",
            tools=[get_weather],
            db=db,
        ),
        Agent(
            id="chat_agent",
            model=Claude(),
            description="You are a concise, friendly general-purpose assistant.",
            db=db,
        ),
    ]


if __name__ == "__main__":
    db = SqliteDb("agent_os.db")
    app = create_app(agents=build_agents(db), db=db)
    serve(app)
    