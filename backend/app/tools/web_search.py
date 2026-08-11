"""Web search tool backed by DuckDuckGo (no API key required).

Mirrors Agno's own quickstart choice of DuckDuckGoTools for the simplest
possible tool-calling example.
"""

from app.tools.base import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for a query and return a summary of top results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # fallback for older package name

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"Error searching for {query!r}: {exc}"

    if not results:
        return f"No results found for {query!r}."

    lines = [f"Search results for {query!r}:"]
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   {href}")
    return "\n".join(lines)
