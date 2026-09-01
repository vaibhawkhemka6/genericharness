"""`SqliteDb` - the one concrete `Db` (`db/base.py`) this stage builds,
mirroring the shape of Agno's `agno/db/sqlite/sqlite.py` (`get_session`/
`upsert_session`) but collapsed from a multi-table, multi-session-type
schema down to the one table this project's `AgentSession`-only world
needs.

Uses the standard library's `sqlite3` directly - no ORM, no migrations
system (`db/migrations/` in real Agno - irrelevant until there are multiple
deployed schema versions to reconcile). One table, one JSON blob column for
the parts of `AgentSession` that don't need to be queried on directly
(`session_data`, `runs`); `session_id`/`agent_id`/`user_id`/`created_at`/
`updated_at` get their own columns since those are exactly the fields
`get_sessions()`'s filters and `get_session()`'s lookup key need indexed/
queryable - `runs` (a whole run history) and `session_data` (free-form) have
no such need, so JSON-in-a-column is the right amount of structure for them.

Per this project's own `.env`/CLAUDE.md convention: Postgres is the eventual
production target, SQLite is what gets built and tested against first (same
reasoning `models/base.py` gives for OpenAI/Claude being the only two
concrete `Model`s built) - every other `db/<provider>/` folder real Agno has
(postgres, mysql, mongo, dynamo, redis, firestore, ...) is out of scope
here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from app.db.base import Db
from app.session.agent import AgentSession

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY,
    agent_id TEXT,
    user_id TEXT,
    session_data TEXT,
    runs TEXT,
    created_at INTEGER,
    updated_at INTEGER
)
"""

_UPSERT_SQL = """
INSERT INTO agent_sessions (session_id, agent_id, user_id, session_data, runs, created_at, updated_at)
VALUES (:session_id, :agent_id, :user_id, :session_data, :runs, :created_at, :updated_at)
ON CONFLICT(session_id) DO UPDATE SET
    agent_id = excluded.agent_id,
    user_id = excluded.user_id,
    session_data = excluded.session_data,
    runs = excluded.runs,
    created_at = excluded.created_at,
    updated_at = excluded.updated_at
"""


class SqliteDb(Db):
    """`Db` backed by a single SQLite file (or `:memory:`) and one
    `agent_sessions` table. Holds one connection open for the instance's
    whole lifetime rather than opening a fresh one per call - not just an
    optimization: a fresh `sqlite3.connect(":memory:")` per call would give
    each call its own *separate*, empty in-memory database (SQLite's
    `:memory:` database lives only as long as the connection that opened
    it), so a smoke test using `SqliteDb(":memory:")` needs the same
    connection to persist across `upsert_session()`/`get_session()` calls
    or every read would see a blank table. File-backed databases don't
    strictly need this, but there's no reason to special-case them.
    """

    def __init__(self, db_file: Union[str, Path] = "agent_sessions.db") -> None:
        self.db_file = str(db_file)
        self._connection = sqlite3.connect(self.db_file, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return self._connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(_CREATE_TABLE_SQL)

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> AgentSession:
        data = {
            "session_id": row["session_id"],
            "agent_id": row["agent_id"],
            "user_id": row["user_id"],
            "session_data": json.loads(row["session_data"]) if row["session_data"] else None,
            "runs": json.loads(row["runs"]) if row["runs"] else [],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        return AgentSession.from_dict(data)

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._row_to_session(row) if row is not None else None

    def get_sessions(
        self,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[AgentSession]:
        clauses = []
        params: List[str] = []
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)

        query = "SELECT * FROM agent_sessions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at ASC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_session(row) for row in rows]

    def upsert_session(self, session: AgentSession) -> AgentSession:
        session_dict = session.to_dict()
        params = {
            "session_id": session_dict["session_id"],
            "agent_id": session_dict["agent_id"],
            "user_id": session_dict["user_id"],
            "session_data": json.dumps(session_dict["session_data"]) if session_dict["session_data"] else None,
            "runs": json.dumps(session_dict["runs"]),
            "created_at": session_dict["created_at"],
            "updated_at": session_dict["updated_at"],
        }
        with self._connect() as connection:
            connection.execute(_UPSERT_SQL, params)
        return session

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM agent_sessions WHERE session_id = ?", (session_id,))
