# GenericHarness — Phase 1

A from-scratch agent harness inspired by [Agno](https://github.com/agno-agi/agno)'s architecture:
a provider-agnostic `Agent` core loop (build context → call model → run tools → respond),
a small `@tool`-decorated tool registry, and a FastAPI backend streamed to a Claude-style
chat UI for testing.

## Phase 1 scope

- Core `Agent` loop with multi-turn tool calling (`backend/app/agent/agent.py`)
- Provider-agnostic `Model` interface with Claude + OpenAI adapters (`backend/app/models/`)
- Tools: `web_search` (DuckDuckGo), `calculator` (safe eval), `get_current_datetime`
- FastAPI backend streaming responses over SSE (`backend/app/main.py`)
- Next.js chat UI with streaming text + visible tool-call blocks (`frontend/`)

Out of scope for Phase 1 (later phases): persistent storage, memory/knowledge, Culture layer,
guardrails, human-in-the-loop approval, multi-agent Teams/Workflows, auth.

## Running it

### 1. Backend

```bash
cd backend
python3.11 -m venv .venv   # or any Python 3.11/3.12 - avoid 3.14, some deps lack wheels for it yet
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in ANTHROPIC_API_KEY and/or OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000
```

Set `DEFAULT_MODEL_PROVIDER` in `.env` to `anthropic` or `openai` to pick which adapter is used.

### 2. Frontend

```bash
cd frontend
npm install   # already installed if you're reading this from the initial scaffold
npm run dev
```

Open the printed localhost URL (usually http://localhost:3000, but Next.js will pick
another port like 3001 if 3000 is already taken by something else on your machine —
the backend's CORS already allows localhost:3000-3002 out of the box).

### 3. Try it

- "What is (180 - 40) / 2?" → exercises the calculator tool
- "What's today's date?" → exercises the datetime tool
- "What's the latest news on X?" → exercises web_search (note: DuckDuckGo's search
  endpoint can rate-limit requests from some cloud/datacenter IPs - if `web_search`
  errors out with a rate-limit message, that's the search backend, not the tool-calling
  logic; swap in a keyed provider like Tavily/Serper later if this is unreliable in
  your deployment environment)

## Project layout

```
backend/
  app/
    agent/       # Agent class + streamed RunEvent types
    models/      # Model interface + Claude/OpenAI adapters
    tools/       # @tool decorator, registry, and the 3 Phase 1 tools
    session/     # in-memory conversation store (per session_id)
    main.py      # FastAPI app, POST /chat (SSE), GET /health
  tests/         # offline agent-loop test using a fake Model (no network)
frontend/
  src/
    app/         # Next.js app router (page.tsx, layout.tsx)
    components/  # Chat.tsx, Message.tsx, ToolCallBlock.tsx
    lib/         # api.ts (SSE client), types.ts
```
