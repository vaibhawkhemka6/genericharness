"""Trivial tool used to sanity-check tool-calling plumbing end to end."""

from datetime import datetime, timezone

from app.tools.base import tool


@tool
def get_current_datetime() -> str:
    """Return the current date and time (UTC) in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC (%A)")
