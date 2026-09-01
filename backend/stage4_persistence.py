"""Stage 5 smoke test: persistence, bottom to top - `AgentSession`
(`app/session/agent.py`) in isolation, `SqliteDb` (`app/db/sqlite/sqlite.py`)
round-tripping it through a real `.db` file, the read/write bookends
(`read_or_create_session()` / `save_session()`, `agent/_storage.py` /
`agent/_session.py`), `get_session_history_messages()`'s three-way
precedence (`agent/_messages.py`), and finally live `Agent.run()` calls
wired end-to-end through `agent/_run.py`'s terminal-path saves.

Follows the same build order as the implementation, and the same "unit
pieces first, then live end-to-end" shape as `stage3b_agent.py`. The one
behavior this stage is *for* - proving live, not just by inspection - gets
its own step at the end: a cancelled run is still persisted (partial
transcript kept, same as Stage 3b's decision for `run()`'s exception
branches), but `AgentSession.get_messages()`'s default `skip_statuses`
means that cancelled run's messages never leak into a later run's context.

`ANTHROPIC_API_KEY` is set, `OPENAI_API_KEY` is blank - same asymmetry as
every earlier stage script; live steps run against `Claude`.

Run: python stage5_persistence.py
"""

from dotenv import load_dotenv

load_dotenv(override=True)

import os
import tempfile

from app.agent._messages import get_session_history_messages
from app.agent._session import save_session
from app.agent._storage import read_or_create_session
from app.agent.agent import Agent
from app.db.sqlite.sqlite import SqliteDb
from app.exceptions import RunCancelledException
from app.models.anthropic.claude import Claude
from app.models.message import Message
from app.run.agent import RunOutput
from app.run.base import RunStatus
from app.session.agent import AgentSession
from app.tools.decorator import tool


def show(n, title):
    print(f"\n{'='*70}\n STEP {n}  {title}\n{'='*70}")


# ══ 1 ══════════════════════════════════════════════════════════════
show(1, "AgentSession in isolation - upsert_run, get_messages, to/from_dict")

s = AgentSession(session_id="sess-1", agent_id="agent-1", user_id="user-1")

run_ok = RunOutput(run_id="r1", status=RunStatus.completed, messages=[
    Message(role="user", content="hi"),
    Message(role="assistant", content="hello!"),
])
run_bad = RunOutput(run_id="r2", status=RunStatus.error, messages=[
    Message(role="user", content="this should be excluded"),
])
s.upsert_run(run_ok)
s.upsert_run(run_bad)
print("runs after 2 upserts:", [r.run_id for r in s.runs])
assert [r.run_id for r in s.runs] == ["r1", "r2"]

# re-upsert r1 with a changed status -> replaced in place, not appended
run_ok_updated = RunOutput(run_id="r1", status=RunStatus.completed, messages=run_ok.messages, content="hello! (updated)")
s.upsert_run(run_ok_updated)
print("runs after re-upsert r1:", [r.run_id for r in s.runs])
assert [r.run_id for r in s.runs] == ["r1", "r2"], "re-upsert must replace, not append"

msgs = s.get_messages()
print("get_messages() default skip_statuses:", [m.content for m in msgs])
assert [m.content for m in msgs] == ["hi", "hello!"], "error-status run must be excluded by default"
assert all(m.from_history for m in msgs), "every returned message must be tagged from_history=True"
assert not run_ok.messages[0].from_history, "the STORED originals must be untouched, not mutated in place"

msgs_all = s.get_messages(skip_statuses=[])
print("get_messages(skip_statuses=[]) includes everything:", [m.content for m in msgs_all])
assert len(msgs_all) == 3

round_tripped = AgentSession.from_dict(s.to_dict())
print("round-tripped runs:", [r.run_id for r in round_tripped.runs])
assert [r.run_id for r in round_tripped.runs] == ["r1", "r2"]
assert round_tripped.runs[0].status == RunStatus.completed
assert round_tripped.runs[1].status == RunStatus.error
print("OK")


# ══ 2 ══════════════════════════════════════════════════════════════
show(2, "SqliteDb - CRUD round trip through a real file-backed db")

tmp_path = os.path.join(tempfile.gettempdir(), "stage5_smoke.db")
if os.path.exists(tmp_path):
    os.remove(tmp_path)
db = SqliteDb(tmp_path)

assert db.get_session("sess-1") is None
db.upsert_session(s)
fetched = db.get_session("sess-1")
print("fetched session_id:", fetched.session_id, "| runs:", [r.run_id for r in fetched.runs])
assert fetched is not None and fetched.session_id == "sess-1"
assert [r.run_id for r in fetched.runs] == ["r1", "r2"]
assert fetched.runs[1].status == RunStatus.error

s2 = AgentSession(session_id="sess-2", agent_id="agent-1", user_id="user-2")
db.upsert_session(s2)
by_agent = db.get_sessions(agent_id="agent-1")
by_user = db.get_sessions(user_id="user-2")
print("get_sessions(agent_id=agent-1):", [x.session_id for x in by_agent])
print("get_sessions(user_id=user-2)  :", [x.session_id for x in by_user])
assert {x.session_id for x in by_agent} == {"sess-1", "sess-2"}
assert [x.session_id for x in by_user] == ["sess-2"]

db.delete_session("sess-2")
assert db.get_session("sess-2") is None
print("OK")


# ══ 3 ══════════════════════════════════════════════════════════════
show(3, "read_or_create_session() / save_session() - the no-db no-op guarantee")

no_db_agent = Agent(model=Claude())
assert read_or_create_session(no_db_agent, "whatever") is None
assert save_session(no_db_agent, s) is s, "save_session must pass through unchanged when agent.db is None"
print("no-db agent: read_or_create_session -> None, save_session -> unchanged passthrough")

db_agent = Agent(model=Claude(), db=db, id="agent-1")
fresh = read_or_create_session(db_agent, "brand-new-session", user_id="user-9")
print("fresh (not yet in db):", fresh.session_id, fresh.agent_id, fresh.user_id, fresh.runs)
assert fresh.session_id == "brand-new-session" and fresh.runs == []
assert db.get_session("brand-new-session") is None, "read_or_create_session must NOT write by itself"

save_session(db_agent, fresh)
assert db.get_session("brand-new-session") is not None
again = read_or_create_session(db_agent, "brand-new-session")
print("second call, now persisted:", again.session_id)
assert again.session_id == "brand-new-session"
print("OK")


# ══ 4 ══════════════════════════════════════════════════════════════
show(4, "get_session_history_messages() three-way precedence, with a real session")

hist_agent = Agent(model=Claude(), db=db, id="agent-1", num_history_messages=10)
real_session = read_or_create_session(hist_agent, "sess-1")  # has r1 (ok) + r2 (error)

from_session = get_session_history_messages(hist_agent, session=real_session)
print("from real session (skip errored run):", [m.content for m in from_session])
assert [m.content for m in from_session] == ["hi", "hello!"]

override = [Message(role="user", content="explicit override wins")]
from_override = get_session_history_messages(hist_agent, session=real_session, session_history=override)
print("explicit session_history overrides the real session:", [m.content for m in from_override])
assert [m.content for m in from_override] == ["explicit override wins"]
print("OK - precedence is session_history > session > agent.session_history stub")


# ══ 5 ══════════════════════════════════════════════════════════════
show(5, "Agent.run() live, db-backed - history round-trips across turns via SqliteDb")

live_db_path = os.path.join(tempfile.gettempdir(), "stage5_live.db")
if os.path.exists(live_db_path):
    os.remove(live_db_path)
live_db = SqliteDb(live_db_path)

memory_agent = Agent(model=Claude(), db=live_db, description="You are concise. Remember what the user tells you.")
sid = "live-session-1"

r1 = memory_agent.run("My favorite color is teal. Just acknowledge it briefly.", session_id=sid)
print("turn 1 content:", r1.content, "| status:", r1.status)
assert r1.status == RunStatus.completed

r2 = memory_agent.run("What's my favorite color? One word answer.", session_id=sid)
print("turn 2 content:", r2.content, "| status:", r2.status)
assert r2.status == RunStatus.completed
assert "teal" in r2.content.lower(), "turn 2 must have seen turn 1's history via the db"

# Confirm it's really coming from storage, not in-process state: a brand-new
# Agent object, same db + session_id, must see the same history.
fresh_agent = Agent(model=Claude(), db=live_db, description="You are concise.")
r3 = fresh_agent.run("What's my favorite color? One word answer.", session_id=sid)
print("turn 3 (new Agent instance, same db+session_id):", r3.content)
assert "teal" in r3.content.lower()
print("OK - history persists across turns AND across separate Agent objects, via the db")


# ══ 6 ══════════════════════════════════════════════════════════════
show(6, "Agent.run() live - a cancelled run is saved, but excluded from future context")


@tool
def force_cancel(reason: str) -> str:
    """Cancel the current run immediately.

    Args:
        reason: Why the run is being cancelled.
    """
    raise RunCancelledException(reason)


sid2 = "live-session-2"
setup_agent = Agent(model=Claude(), db=live_db, description="You are concise.")
setup_agent.run("My favorite color is teal. Just acknowledge it briefly.", session_id=sid2)

cancel_agent = Agent(
    model=Claude(),
    db=live_db,
    description="Always call the force_cancel tool first, with any reason, before doing anything else.",
    tools=[force_cancel],
)
r_cancel = cancel_agent.run(
    "Ignore everything before this. My favorite color is actually purple - call force_cancel now.",
    session_id=sid2,
)
print("cancelled run status:", r_cancel.status, "| content:", r_cancel.content)
assert r_cancel.status == RunStatus.cancelled

# The cancelled run's messages ARE on disk (partial transcript kept)...
persisted = live_db.get_session(sid2)
statuses = [r.status for r in persisted.runs]
print("persisted run statuses:", statuses)
assert RunStatus.cancelled in statuses, "cancelled run must still be persisted, not dropped"

# ...but get_messages()'s default skip_statuses keeps it out of replayed context.
replay = persisted.get_messages()
replay_text = " ".join(str(m.content) for m in replay)
print("replayed history text:", replay_text)
assert "purple" not in replay_text.lower(), "cancelled run's messages must not leak into future context"
assert "teal" in replay_text.lower()

# Prove it live: the next turn must still answer "teal", never having "seen" purple.
followup_agent = Agent(model=Claude(), db=live_db, description="You are concise. One word answers.")
r_followup = followup_agent.run("What's my favorite color?", session_id=sid2)
print("follow-up run content:", r_followup.content)
assert "teal" in r_followup.content.lower()
assert "purple" not in r_followup.content.lower()
print("OK - cancelled runs are persisted but never replayed as context")


print("\nALL STAGE 5 PERSISTENCE STEPS OK")
