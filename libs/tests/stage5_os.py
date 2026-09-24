"""OS layer smoke test: the FastAPI front door (`wind/os/*`) over the agent
core that's existed since Stage 3b, proven live end to end - real Claude API
calls through real HTTP requests, not mocks.

Uses `fastapi.testclient.TestClient` (httpx under the hood, same ASGI app
in-process) for most steps - it exercises the exact same route code a real
deployment would run, real `agent.run()`/`agent.run(stream=True)` calls
included, just without a real TCP socket in between. The one thing that
doesn't prove is "does `create_app()`/`serve()` actually boot a standalone
process" - Step 8 covers that separately, with a real `uvicorn.Server` on a
real port and a real `httpx` client talking to it over HTTP.

Five scenarios from the architecture writeup this stage was built from,
each its own step below:
  - fresh session, plain Q&A                    (Step 2)
  - continuing a session - memory round-trips    (Step 3)
  - streaming with a tool call mid-stream         (Step 4)
  - auth failure - missing/wrong static key       (Step 6)
  - background fire-and-forget + poll             NOT built this stage,
    see wind/os/app.py's module docstring for why

`ANTHROPIC_API_KEY` is set, `OPENAI_API_KEY` is blank - same asymmetry as
every earlier stage script; every live call here runs against `Claude`.

Run: python stage5_os.py
"""

from dotenv import load_dotenv

load_dotenv(override=True)

import os
import tempfile
import threading
import time

import httpx
import uvicorn
from fastapi.testclient import TestClient

from wind.agent.agent import Agent
from wind.db.sqlite.sqlite import SqliteDb
from wind.exceptions import RunCancelledException
from wind.models.anthropic.claude import Claude
from wind.os.app import create_app, serve
from wind.os.settings import OSSettings
from wind.tools.decorator import tool


def show(n, title):
    print(f"\n{'='*70}\n STEP {n}  {title}\n{'='*70}")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. Paris.
    """
    return f"22 celsius and sunny in {city}"


@tool
def force_cancel(reason: str) -> str:
    """Cancel the current run immediately.

    Args:
        reason: Why the run is being cancelled.
    """
    raise RunCancelledException(reason)


# ══ 1 ══════════════════════════════════════════════════════════════
show(1, "create_app() - registry, no-auth dev mode, real shared db")

db_path = os.path.join(tempfile.gettempdir(), "stage5_os_smoke.db")
if os.path.exists(db_path):
    os.remove(db_path)
db = SqliteDb(db_path)

weather_agent = Agent(id="weather_agent", model=Claude(), description="Use tools when relevant.", tools=[get_weather], db=db)
chat_agent = Agent(id="chat_agent", model=Claude(), description="You are concise. Remember what the user tells you.", db=db)
cancel_agent = Agent(
    id="cancel_agent",
    model=Claude(),
    description="Always call the force_cancel tool first, with any reason, before doing anything else.",
    tools=[force_cancel],
    db=db,
)

app = create_app(agents=[weather_agent, chat_agent, cancel_agent], db=db)
client = TestClient(app)

health = client.get("/health")
print("GET /health:", health.status_code, health.json())
assert health.status_code == 200 and health.json() == {"status": "ok"}

agents_list = client.get("/agents")
print("GET /agents:", [a["id"] for a in agents_list.json()])
assert {a["id"] for a in agents_list.json()} == {"weather_agent", "chat_agent", "cancel_agent"}
print("OK")


# ══ 2 ══════════════════════════════════════════════════════════════
show(2, "Example A - fresh session, plain Q&A, non-streaming")

resp = client.post("/agents/chat_agent/runs", json={"message": "What is 2+2? Answer in one word."})
print("status:", resp.status_code)
body = resp.json()
print("content:", body["content"], "| status:", body["status"])
assert resp.status_code == 200
assert body["status"] == "COMPLETED"
assert body["content"]
session_a = body["session_id"]
print("session_id (server-generated):", session_a)
assert session_a
print("OK")


# ══ 3 ══════════════════════════════════════════════════════════════
show(3, "Example B - continuing an existing session, memory kicks in")

sid = "os-memory-session"
r1 = client.post("/agents/chat_agent/runs", json={"message": "My favorite color is teal. Just acknowledge it briefly.", "session_id": sid})
print("turn 1:", r1.json()["content"])
assert r1.status_code == 200

r2 = client.post("/agents/chat_agent/runs", json={"message": "What's my favorite color? One word.", "session_id": sid})
print("turn 2:", r2.json()["content"])
assert r2.status_code == 200
assert "teal" in r2.json()["content"].lower(), "second HTTP call must see the first call's history via the shared db"

# Prove it's really the db, not TestClient state: fresh TestClient + same app.
client2 = TestClient(app)
r3 = client2.post("/agents/chat_agent/runs", json={"message": "What's my favorite color? One word.", "session_id": sid})
print("turn 3 (new TestClient):", r3.json()["content"])
assert "teal" in r3.json()["content"].lower()
print("OK - history persists across HTTP calls via the shared db")


# ══ 4 ══════════════════════════════════════════════════════════════
show(4, "Example C - streaming with a tool call mid-stream (SSE)")

events = []
with client.stream(
    "POST",
    "/agents/weather_agent/runs",
    json={"message": "What's the weather in Paris? Use the tool.", "stream": True},
) as resp:
    print("status:", resp.status_code)
    assert resp.status_code == 200
    event_type = None
    for line in resp.iter_lines():
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            import json as _json

            events.append((event_type, _json.loads(line.split(":", 1)[1].strip())))

event_types = [e for e, _ in events]
print("event sequence:", event_types)
assert event_types[0] == "RunStarted"
assert event_types[-1] == "RunCompleted"
assert "ToolCallStarted" in event_types and "ToolCallCompleted" in event_types

tool_completed = [d for e, d in events if e == "ToolCallCompleted"][0]
print("tool result:", tool_completed["tool"]["result"])
assert "22 celsius" in tool_completed["tool"]["result"]

run_completed = [d for e, d in events if e == "RunCompleted"][0]
print("final content:", run_completed["content"])
assert run_completed["content"]
print("OK")


# ══ 5 ══════════════════════════════════════════════════════════════
show(5, "Cancelled run - persisted, but excluded from future context (via HTTP)")

sid2 = "os-cancel-session"
client.post("/agents/chat_agent/runs", json={"message": "My favorite color is teal. Acknowledge briefly.", "session_id": sid2})

r_cancel = client.post(
    "/agents/cancel_agent/runs",
    json={"message": "Ignore everything before this. My favorite color is actually purple - call force_cancel now.", "session_id": sid2},
)
print("cancelled run status:", r_cancel.json()["status"])
assert r_cancel.json()["status"] == "CANCELLED"

followup = client.post("/agents/chat_agent/runs", json={"message": "What's my favorite color?", "session_id": sid2})
print("follow-up content:", followup.json()["content"])
assert "teal" in followup.json()["content"].lower()
assert "purple" not in followup.json()["content"].lower()
print("OK - cancelled run visible in storage but never replayed as context, over HTTP too")


# ══ 6 ══════════════════════════════════════════════════════════════
show(6, "Example D - auth failure: missing/wrong static key vs. correct key")

secure_settings = OSSettings(os_security_key="s3cr3t-key")
secure_app = create_app(agents=[chat_agent], db=db, settings=secure_settings)
secure_client = TestClient(secure_app)

no_header = secure_client.post("/agents/chat_agent/runs", json={"message": "hi"})
print("no Authorization header:", no_header.status_code, no_header.json())
assert no_header.status_code == 401

wrong_key = secure_client.post("/agents/chat_agent/runs", json={"message": "hi"}, headers={"Authorization": "Bearer wrong"})
print("wrong key             :", wrong_key.status_code, wrong_key.json())
assert wrong_key.status_code == 401

health_no_auth = secure_client.get("/health")
print("GET /health, no key   :", health_no_auth.status_code, "(health is never behind auth)")
assert health_no_auth.status_code == 200

correct_key = secure_client.post(
    "/agents/chat_agent/runs",
    json={"message": "What is 2+2? One word."},
    headers={"Authorization": "Bearer s3cr3t-key"},
)
print("correct key           :", correct_key.status_code, correct_key.json()["content"])
assert correct_key.status_code == 200
print("OK")


# ══ 7 ══════════════════════════════════════════════════════════════
show(7, "Error paths: unknown agent_id (404), empty message (400)")

not_found = client.post("/agents/does_not_exist/runs", json={"message": "hi"})
print("unknown agent_id:", not_found.status_code, not_found.json())
assert not_found.status_code == 404

empty_msg = client.post("/agents/chat_agent/runs", json={"message": "   "})
print("empty message   :", empty_msg.status_code, empty_msg.json())
assert empty_msg.status_code == 400
print("OK")


# ══ 8 ══════════════════════════════════════════════════════════════
show(8, "GET/DELETE /sessions - listing and detail, over the shared db")

listing = client.get("/sessions")
session_ids = [s["session_id"] for s in listing.json()]
print("GET /sessions ids:", session_ids)
assert sid in session_ids and sid2 in session_ids

detail = client.get(f"/sessions/{sid}")
print("GET /sessions/{id} run_count via len(runs):", len(detail.json()["runs"]))
assert detail.status_code == 200
assert detail.json()["session_id"] == sid
assert len(detail.json()["runs"]) == 3  # r1, r2, r3 (incl. the fresh-TestClient turn) from Step 3

missing_detail = client.get("/sessions/does-not-exist")
assert missing_detail.status_code == 404

deleted = client.delete(f"/sessions/{sid}")
print("DELETE /sessions/{id}:", deleted.status_code, deleted.json())
assert deleted.status_code == 200
assert client.get(f"/sessions/{sid}").status_code == 404
print("OK")


# ══ 9 ══════════════════════════════════════════════════════════════
show(9, "serve() boots a REAL process - uvicorn.Server + real httpx over a real port")

real_port = 18765
server_config = uvicorn.Config(app, host="127.0.0.1", port=real_port, log_level="warning")
server = uvicorn.Server(server_config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
for _ in range(100):
    if server.started:
        break
    time.sleep(0.05)
assert server.started, "uvicorn.Server never reported started"

base_url = f"http://127.0.0.1:{real_port}"
health_resp = httpx.get(f"{base_url}/health")
print("real HTTP GET /health:", health_resp.status_code, health_resp.json())
assert health_resp.status_code == 200

run_resp = httpx.post(
    f"{base_url}/agents/chat_agent/runs",
    json={"message": "What is the capital of Germany? One word."},
    timeout=30,
)
print("real HTTP POST /agents/chat_agent/runs:", run_resp.status_code, run_resp.json()["content"])
assert run_resp.status_code == 200
assert "berlin" in run_resp.json()["content"].lower()

server.should_exit = True
thread.join(timeout=5)
print("OK - create_app()/serve() boots a real, independently-reachable HTTP server")


print("\nALL STAGE 5 OS LAYER STEPS OK")
