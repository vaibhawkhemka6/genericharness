"""Central config loaded from environment variables."""
import os

from dotenv import dotenv_values, load_dotenv

# override=True: values in .env win even if an empty/stale var of the same
# name already exists in the process environment (some shells/sandboxes
# pre-set keys like ANTHROPIC_API_KEY="" which would otherwise silently
# shadow the real value from .env, since load_dotenv() defaults to not
# overriding existing env vars).
load_dotenv(override=True)

# Values that exist *only* in backend/.env, ignoring the ambient process
# environment entirely. Used for a couple of settings below where inheriting
# an unrelated ambient value (e.g. a sandbox/CI pre-setting ANTHROPIC_BASE_URL)
# would silently break things rather than just being redundant.
_dotenv_only = dotenv_values()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Only trust a custom Anthropic base URL if it's explicitly set in our own
# backend/.env - never inherit one from the ambient shell/sandbox
# environment. Some environments pre-set ANTHROPIC_BASE_URL to a value that
# already includes "/v1", which makes the SDK double it up into
# ".../v1/v1/messages" and every request 404s.
ANTHROPIC_BASE_URL = _dotenv_only.get("ANTHROPIC_BASE_URL") or ""
OPENAI_BASE_URL = _dotenv_only.get("OPENAI_BASE_URL") or ""

DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "anthropic")
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "claude-sonnet-4-5")

# Comma-separated list. Next.js picks the next free port (3000, 3001, ...) if
# 3000 is already taken by something else on your machine, so we allow a
# small range of local dev ports out of the box.
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGIN", "http://localhost:3000,http://localhost:3001,http://localhost:3002"
    ).split(",")
    if origin.strip()
]
