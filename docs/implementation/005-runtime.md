# 005: Agent Runtime — The Deployment Shell

**Status:** Ready to Build
**Priority:** P0 — Makes Nous actually runnable
**Estimated Effort:** 8-10 hours
**Prerequisites:** 001 (merged), 002 (merged), 003 (merged), 004-cognitive-layer (in progress)
**Feature Spec:** [F004-runtime.md](../features/F004-runtime.md)

## Objective

Implement the Runtime — the deployment shell that packages Brain, Heart, and Cognitive Layer into a runnable agent. One `docker compose up` gives you a thinking agent with REST API, MCP external interface, and persistent memory.

After this phase:
```bash
docker compose up -d
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Should we use Redis for caching?"}'
```

## Architecture

```
┌──────────────────────────────────────────────┐
│              Nous Container                   │
│                                               │
│  main.py (entry point)                        │
│    ├── Settings (config.py)                   │
│    ├── Database (storage/)                    │
│    ├── Brain (brain/)                         │
│    ├── Heart (heart/)                         │
│    ├── CognitiveLayer (cognitive/)            │
│    └── App (api/)                             │
│         ├── REST routes (rest.py)             │
│         ├── MCP server (mcp.py)               │
│         └── Agent runner (runner.py)          │
│                                               │
│  Starlette ASGI ──► Uvicorn                   │
└──────────────────┬───────────────────────────┘
                   │ :8000 REST + MCP
              ┌────▼─────┐
              │ Postgres  │
              └──────────┘
```

## File-by-File Specification

### 1. `nous/config.py` — Settings Updates (~20 lines added)

Add runtime-specific settings to the existing Settings class.

```python
# Add to existing Settings class:

    # Runtime
    host: str = "0.0.0.0"
    port: int = 8000
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")

    # Agent identity
    agent_name: str = "Nous"
    agent_description: str = "A thinking agent that learns from experience"
    identity_prompt: str = ""  # Overridden by config file or env

    # MCP
    mcp_enabled: bool = True

    # Model
    model: str = "claude-sonnet-4-5-20250514"
    max_tokens: int = 4096
```

### 2. `nous/api/__init__.py` (~5 lines)

```python
"""Nous API — REST and MCP interfaces."""
```

### 3. `nous/api/runner.py` (~150 lines)

The agent runner — wraps the LLM call with the Cognitive Layer.

```python
"""Agent runner — executes a single conversational turn.

Wires CognitiveLayer.pre_turn() and post_turn() around
the actual LLM call. This is the core execution loop.
"""

import logging
from dataclasses import dataclass, field

import httpx

from nous.brain import Brain
from nous.cognitive import CognitiveLayer, TurnContext, TurnResult
from nous.config import Settings
from nous.heart import Heart

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """Tracks a multi-turn conversation."""
    session_id: str
    messages: list[Message] = field(default_factory=list)
    turn_contexts: list[TurnContext] = field(default_factory=list)


class AgentRunner:
    """Runs conversational turns with cognitive layer hooks.

    For v0.1.0, uses the Anthropic Messages API directly via httpx.
    Claude Agent SDK integration is a future enhancement — the cognitive
    layer's pre_turn/post_turn design is SDK-ready but doesn't require it.
    """

    def __init__(
        self,
        cognitive: CognitiveLayer,
        settings: Settings,
    ) -> None:
        self._cognitive = cognitive
        self._settings = settings
        self._conversations: dict[str, Conversation] = {}
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize the HTTP client for Anthropic API."""
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": self._settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def run_turn(
        self,
        session_id: str,
        user_message: str,
        agent_id: str | None = None,
    ) -> tuple[str, TurnContext]:
        """Execute a single conversational turn.

        Steps:
        1. Get or create Conversation for session_id
        2. Call cognitive.pre_turn() → TurnContext (system prompt, frame, decision_id)
        3. Append user message to conversation history
        4. Call Anthropic Messages API with:
           - system: turn_context.system_prompt
           - messages: conversation history
           - model: settings.model
           - max_tokens: settings.max_tokens
        5. Extract assistant response text
        6. Append assistant message to conversation history
        7. Call cognitive.post_turn() with TurnResult
        8. Return (response_text, turn_context)

        On API error:
        - Log the error
        - Create TurnResult with error field set
        - Still call post_turn (so monitor can assess)
        - Return error message to user
        """
        _agent_id = agent_id or self._settings.agent_id

    async def end_conversation(self, session_id: str, agent_id: str | None = None) -> None:
        """End a conversation with reflection.

        1. If conversation has >= 3 turns, generate reflection:
           - Call Anthropic API with conversation history + prompt:
             "Review this conversation. What was learned? List key facts as
              'learned: <fact>' lines. What went well? What should be done
              differently next time?"
           - This is the ONLY place the cognitive layer receives LLM-generated
             reflection. CognitiveLayer.end_session() itself makes no LLM calls.
        2. Call cognitive.end_session(reflection=reflection_text)
        3. Remove from self._conversations
        """

    def _get_or_create_conversation(self, session_id: str) -> Conversation:
        """Get existing or create new conversation."""
        if session_id not in self._conversations:
            self._conversations[session_id] = Conversation(session_id=session_id)
        return self._conversations[session_id]

    def _format_messages(self, conversation: Conversation) -> list[dict]:
        """Format conversation history for Anthropic API.

        Returns list of {"role": "user"|"assistant", "content": "..."}
        Limits to last 20 messages to avoid token overflow.
        """
```

**Implementation notes:**
- Direct httpx call to `POST /v1/messages` — no SDK dependency for v0.1.0
- Conversation history kept in memory (dict keyed by session_id)
- History capped at last 20 messages (configurable later)
- pre_turn system prompt is rebuilt each turn (context may refresh)

### 4. `nous/api/rest.py` (~250 lines)

REST API via Starlette.

```python
"""REST API for Nous agent.

Endpoints:
  POST /chat              — Send message, get response
  DELETE /chat/{session}  — End conversation
  GET  /status            — Agent status + memory stats + calibration
  GET  /decisions         — List recent decisions (Brain)
  GET  /decisions/{id}    — Get decision detail
  GET  /episodes          — List recent episodes (Heart)
  GET  /facts             — Search facts (Heart)
  GET  /censors           — Active censors (Heart)
  GET  /frames            — Available frames
  GET  /calibration       — Calibration report (Brain)
  GET  /health            — Health check (DB connectivity)
"""

import logging
from uuid import uuid4

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from nous.api.runner import AgentRunner
from nous.brain import Brain
from nous.cognitive import CognitiveLayer
from nous.config import Settings
from nous.heart import Heart
from nous.storage.database import Database

logger = logging.getLogger(__name__)


def create_app(
    runner: AgentRunner,
    brain: Brain,
    heart: Heart,
    cognitive: CognitiveLayer,
    database: Database,
    settings: Settings,
) -> Starlette:
    """Create the Starlette ASGI app with all routes."""

    async def chat(request: Request) -> JSONResponse:
        """POST /chat — Send a message, get a response.

        Body: {"message": "...", "session_id": "..." (optional)}
        Response: {
            "response": "...",
            "session_id": "...",
            "frame": "...",
            "decision_id": "..." or null
        }

        If session_id not provided, generates a UUID.
        """

    async def end_chat(request: Request) -> JSONResponse:
        """DELETE /chat/{session_id} — End a conversation.

        Calls runner.end_conversation().
        Response: {"status": "ended", "session_id": "..."}
        """

    async def status(request: Request) -> JSONResponse:
        """GET /status — Agent status overview.

        Response: {
            "agent_id": "...",
            "agent_name": "...",
            "model": "...",
            "calibration": { brierScore, accuracy, totalDecisions, reviewedCount },
            "memory": {
                "active_conversations": count,
                "active_censors": count,
                "total_decisions": count,
                "total_facts": count,
                "total_episodes": count,
                "total_procedures": count
            }
        }

        Calibration via Brain.get_calibration().
        Memory counts via simple SELECT COUNT(*) queries.
        """

    async def list_decisions(request: Request) -> JSONResponse:
        """GET /decisions?limit=20&offset=0 — Recent decisions.

        Query params: limit (default 20), offset (default 0)
        Response: {"decisions": [...], "total": count}
        Uses Brain.query() with empty string for broad match,
        or a simple SQL query ordered by created_at DESC.
        """

    async def get_decision(request: Request) -> JSONResponse:
        """GET /decisions/{id} — Decision detail.

        Response: full DecisionDetail as dict.
        Uses Brain.get(id).
        """

    async def list_episodes(request: Request) -> JSONResponse:
        """GET /episodes?limit=20 — Recent episodes.

        Uses Heart.episodes.search() with empty query.
        """

    async def search_facts(request: Request) -> JSONResponse:
        """GET /facts?q=query&limit=20 — Search facts.

        Query param q required.
        Uses Heart.facts.search(q).
        """

    async def list_censors(request: Request) -> JSONResponse:
        """GET /censors — Active censors.

        Uses Heart.censors.search(agent_id) filtered to status=active.
        """

    async def list_frames(request: Request) -> JSONResponse:
        """GET /frames — Available cognitive frames.

        Uses CognitiveLayer._frames.list_frames(agent_id).
        """

    async def calibration(request: Request) -> JSONResponse:
        """GET /calibration — Full calibration report.

        Uses Brain.get_calibration(agent_id).
        """

    async def health(request: Request) -> JSONResponse:
        """GET /health — Health check.

        Tests DB connectivity with a simple SELECT 1.
        Response: {"status": "healthy"} or {"status": "unhealthy", "error": "..."}
        """

    routes = [
        Route("/chat", chat, methods=["POST"]),
        Route("/chat/{session_id}", end_chat, methods=["DELETE"]),
        Route("/status", status),
        Route("/decisions", list_decisions),
        Route("/decisions/{id}", get_decision),
        Route("/episodes", list_episodes),
        Route("/facts", search_facts),
        Route("/censors", list_censors),
        Route("/frames", list_frames),
        Route("/calibration", calibration),
        Route("/health", health),
    ]

    return Starlette(routes=routes)
```

**Implementation notes:**
- Starlette over FastAPI — lighter, fewer dependencies, sufficient for this use case
- All endpoints return JSONResponse
- Error handling: try/except around each handler, return 500 with error message
- CORS not needed for v0.1.0 (add later if dashboard is separate origin)

### 5. `nous/api/mcp.py` (~180 lines)

MCP server for external agent-to-agent communication.

```python
"""MCP interface — lets other agents interact with Nous.

Exposes 5 tools:
  nous_chat    — Send a message, get a response
  nous_recall  — Search across all memory types
  nous_status  — Get agent status and calibration
  nous_teach   — Add a fact or procedure to Nous's memory
  nous_decide  — Ask Nous to make a decision (forces decision frame)

Uses mcp library's low-level Server + Streamable HTTP transport,
mounted at /mcp on the same Starlette app.
"""

import logging
from typing import Any

from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool

from nous.api.runner import AgentRunner
from nous.brain import Brain
from nous.config import Settings
from nous.heart import Heart

logger = logging.getLogger(__name__)


def create_mcp_server(
    runner: AgentRunner,
    brain: Brain,
    heart: Heart,
    settings: Settings,
) -> StreamableHTTPSessionManager:
    """Create MCP server with Nous tools.

    Returns StreamableHTTPSessionManager to be mounted on Starlette.
    """

    server = Server("nous")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="nous_chat",
                description="Send a message to Nous and get a response. Nous will think about your message using its decision intelligence and memory systems.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Your message to Nous"},
                        "session_id": {"type": "string", "description": "Optional session ID for multi-turn conversations"},
                    },
                    "required": ["message"],
                },
            ),
            Tool(
                name="nous_recall",
                description="Search Nous's memory for relevant information. Searches across decisions, facts, episodes, and procedures.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["all", "decisions", "facts", "episodes", "procedures"],
                            "description": "Type of memory to search (default: all)",
                        },
                        "limit": {"type": "integer", "description": "Max results (default: 5)"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="nous_status",
                description="Get Nous's current status: calibration accuracy, memory counts, active frames.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="nous_teach",
                description="Teach Nous a new fact or procedure. Facts are things to know; procedures are how-to knowledge.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["fact", "procedure"],
                            "description": "What kind of knowledge",
                        },
                        "content": {"type": "string", "description": "The knowledge to teach"},
                        "domain": {"type": "string", "description": "Domain or category (optional)"},
                        "source": {"type": "string", "description": "Where this knowledge came from"},
                    },
                    "required": ["type", "content"],
                },
            ),
            Tool(
                name="nous_decide",
                description="Ask Nous to make a decision. Forces the decision cognitive frame for thorough analysis with guardrails, similar past decisions, and calibrated confidence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The decision to make"},
                        "stakes": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "How important is this decision (default: medium)",
                        },
                    },
                    "required": ["question"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Route tool calls to appropriate handlers.

        nous_chat → runner.run_turn()
        nous_recall → brain.query() + heart.recall()
        nous_status → brain.get_calibration() + memory counts
        nous_teach → heart.facts.learn() or heart.procedures.store()
        nous_decide → runner.run_turn() with forced decision frame
        """

    # Create session manager
    session_manager = StreamableHTTPSessionManager(server)
    return session_manager
```

**Implementation notes:**
- MCP server mounted at `/mcp` on the Starlette app
- `nous_decide` forces the decision frame by prepending "Should we: " to the input if not already a question
- `nous_recall` with type="all" searches brain.query + heart.recall and merges results
- `nous_teach` creates facts with source="mcp:{caller}" and procedures with domain from arg
- Session manager handles SSE transport automatically

### 6. `nous/main.py` (~120 lines)

Entry point. Wires everything together.

```python
"""Nous agent entry point.

Initializes all components and starts the server:
  Settings → Database → Brain → Heart → CognitiveLayer → Runner → App → Uvicorn
"""

import asyncio
import logging
import sys

import uvicorn
from starlette.routing import Mount

from nous.api.mcp import create_mcp_server
from nous.api.rest import create_app
from nous.api.runner import AgentRunner
from nous.brain import Brain
from nous.brain.embeddings import EmbeddingProvider
from nous.cognitive import CognitiveLayer
from nous.config import Settings
from nous.heart import Heart
from nous.storage.database import Database

logger = logging.getLogger(__name__)


async def create_components(settings: Settings) -> tuple:
    """Initialize all components in dependency order.

    Returns (database, brain, heart, cognitive, runner).

    1. Database — connection pool
    2. EmbeddingProvider — optional (None if no API key)
    3. Brain — decision intelligence
    4. Heart — memory system
    5. CognitiveLayer — orchestrator
    6. AgentRunner — LLM integration
    """
    database = Database(settings)
    await database.initialize()

    embedding_provider = None
    if settings.openai_api_key:
        embedding_provider = EmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    brain = Brain(database, settings, embedding_provider)
    heart = Heart(database, settings, embedding_provider)
    cognitive = CognitiveLayer(brain, heart, settings, settings.identity_prompt)
    runner = AgentRunner(cognitive, settings)
    await runner.start()

    return database, brain, heart, cognitive, runner


async def shutdown(database, brain, heart, runner) -> None:
    """Graceful shutdown in reverse order."""
    logger.info("Shutting down Nous...")
    await runner.close()
    await heart.close()
    await brain.close()
    await database.close()
    logger.info("Nous shutdown complete.")


def build_app(
    runner: AgentRunner,
    brain: Brain,
    heart: Heart,
    cognitive: CognitiveLayer,
    database: Database,
    settings: Settings,
) -> object:
    """Build the combined Starlette app with REST + MCP.

    REST routes mounted at /
    MCP server mounted at /mcp
    """
    app = create_app(runner, brain, heart, cognitive, database, settings)

    if settings.mcp_enabled:
        mcp_manager = create_mcp_server(runner, brain, heart, settings)
        # Mount MCP routes
        app.routes.append(Mount("/mcp", app=mcp_manager.handle_request))

    return app


def main() -> None:
    """Entry point — parse settings, build app, run server."""
    settings = Settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Starting Nous agent: %s (%s)", settings.agent_name, settings.agent_id)
    logger.info("Model: %s", settings.model)
    logger.info("Database: %s:%s/%s", settings.db_host, settings.db_port, settings.db_name)
    logger.info("MCP: %s", "enabled" if settings.mcp_enabled else "disabled")

    # Build components synchronously for uvicorn
    loop = asyncio.new_event_loop()
    database, brain, heart, cognitive, runner = loop.run_until_complete(
        create_components(settings)
    )
    loop.close()

    app = build_app(runner, brain, heart, cognitive, database, settings)

    # Register shutdown
    import atexit
    atexit.register(lambda: asyncio.run(shutdown(database, brain, heart, runner)))

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
```

### 7. `Dockerfile` (~25 lines)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install psql client for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[runtime]"

# Copy source
COPY nous/ nous/
COPY sql/ sql/

HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "nous.main"]
```

### 8. `docker-compose.yml` — Updated (~50 lines)

Replace the existing single-service compose with the full stack.

```yaml
services:
  nous:
    build: .
    container_name: nous-agent
    ports:
      - "${NOUS_PORT:-8000}:8000"
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_USER=nous
      - DB_PASSWORD=${DB_PASSWORD:-nous_dev_password}
      - DB_NAME=nous
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - NOUS_AGENT_ID=${NOUS_AGENT_ID:-nous-default}
      - NOUS_AGENT_NAME=${NOUS_AGENT_NAME:-Nous}
      - NOUS_MODEL=${NOUS_MODEL:-claude-sonnet-4-5-20250514}
      - NOUS_LOG_LEVEL=${NOUS_LOG_LEVEL:-info}
      - NOUS_MCP_ENABLED=${NOUS_MCP_ENABLED:-true}
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg17
    container_name: nous-postgres
    environment:
      POSTGRES_DB: nous
      POSTGRES_USER: nous
      POSTGRES_PASSWORD: ${DB_PASSWORD:-nous_dev_password}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/001-init.sql
      - ./sql/seed.sql:/docker-entrypoint-initdb.d/002-seed.sql
    ports:
      - "${DB_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nous"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### 9. `pyproject.toml` — Updated dependencies

Add a `[project.optional-dependencies]` runtime group:

```toml
[project.optional-dependencies]
runtime = [
    "uvicorn>=0.30,<1.0",
    "starlette>=0.40,<1.0",
    "mcp>=1.0,<2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
]
```

### 10. `.env.example` — Updated

```env
# DB connection — shared by docker-compose AND Python app
DB_HOST=localhost
DB_PORT=5432
DB_USER=nous
DB_PASSWORD=nous_dev_password
DB_NAME=nous

# LLM
ANTHROPIC_API_KEY=your_anthropic_key_here

# Embeddings (optional — keyword-only search if omitted)
OPENAI_API_KEY=your_openai_key_here

# Agent
NOUS_AGENT_ID=nous-default
NOUS_AGENT_NAME=Nous
NOUS_MODEL=claude-sonnet-4-5-20250514
NOUS_LOG_LEVEL=info
NOUS_MCP_ENABLED=true
NOUS_PORT=8000
```

## Test Specification

### `tests/test_runner.py` (~180 lines)

```python
# Uses a mock HTTP server (httpx.MockTransport) to simulate Anthropic API.

# test_run_turn_basic — sends message, gets response, returns (text, context)
# test_run_turn_creates_conversation — new session_id creates Conversation
# test_run_turn_appends_messages — conversation history grows each turn
# test_run_turn_calls_pre_turn — cognitive.pre_turn() called with input
# test_run_turn_calls_post_turn — cognitive.post_turn() called with result
# test_run_turn_api_error — API 500 → error message returned, post_turn still called
# test_run_turn_history_capped — messages capped at 20
# test_end_conversation — removes from dict, calls cognitive.end_session()
# test_end_conversation_nonexistent — doesn't error

# Mock fixture: MockCognitiveLayer that returns preset TurnContext
# Mock fixture: httpx.MockTransport returning {"content": [{"text": "response"}]}
```

### `tests/test_rest.py` (~220 lines)

```python
# Uses Starlette TestClient (httpx-based, no server needed).

# test_chat_basic — POST /chat → 200, response has message + session_id + frame
# test_chat_with_session — POST /chat with session_id → same session reused
# test_end_chat — DELETE /chat/{session} → 200
# test_status — GET /status → agent info + calibration + memory counts
# test_list_decisions — GET /decisions → list with total
# test_get_decision — GET /decisions/{id} → detail
# test_get_decision_not_found — GET /decisions/{bad_id} → 404
# test_list_episodes — GET /episodes → list
# test_search_facts — GET /facts?q=test → results
# test_search_facts_no_query — GET /facts → 400
# test_list_censors — GET /censors → active censors
# test_list_frames — GET /frames → 6 frames
# test_calibration — GET /calibration → report
# test_health — GET /health → healthy
# test_health_db_down — GET /health with broken DB → unhealthy

# Fixtures: pre-seeded DB with decisions, facts, episodes, censors
# Fixture: MockAgentRunner that returns canned responses
```

### `tests/test_mcp.py` (~150 lines)

```python
# Tests MCP tool handlers directly (call_tool function).

# test_nous_chat — calls runner.run_turn, returns response
# test_nous_recall_all — searches brain + heart, merges results
# test_nous_recall_decisions — brain.query only
# test_nous_recall_facts — heart.facts.search only
# test_nous_status — returns calibration + counts
# test_nous_teach_fact — creates fact via heart.facts.learn
# test_nous_teach_procedure — creates procedure via heart.procedures.store
# test_nous_decide — forces decision frame, calls runner.run_turn

# Fixtures: pre-seeded DB, MockAgentRunner
```

## Key Design Decisions

### D1: Starlette over FastAPI
Less magic, fewer dependencies, sufficient for our use case. We don't need automatic OpenAPI docs or Pydantic request validation (we do it manually). Easier to mount MCP alongside REST.

### D2: Direct Anthropic API via httpx (no SDK for v0.1.0)
The Claude Agent SDK adds tool use, streaming, and hooks — but we're implementing our own hook system (CognitiveLayer). Using the Messages API directly keeps the dependency surface small. SDK integration is a future enhancement when we add tool use.

### D3: In-memory conversation state
Conversations live in a dict, not in the database. This is fine for v0.1.0 (single-instance). Persistent conversation state is a future feature. Episode tracking in Heart already captures the important bits.

### D4: Combined REST + MCP on single port
One port (8000) serves both REST (/) and MCP (/mcp). Simpler deployment than separate ports. MCP clients connect to `http://host:8000/mcp`.

### D5: Graceful shutdown via atexit
Not perfect (doesn't handle SIGKILL), but sufficient for Docker's SIGTERM → 10s grace. Uvicorn handles SIGTERM; atexit runs on clean exit.

### D6: MCP `nous_decide` forces decision frame
When another agent asks Nous to decide something, it should always use the full decision machinery (guardrails, similar decisions, deliberation). The decision frame is forced regardless of input text pattern matching.

### D7: No authentication for v0.1.0
The REST API and MCP server have no auth. This is a local/dev deployment. Auth is a future feature (API keys, JWT, or MCP auth extension).

### D8: Runtime dependencies are optional
`uvicorn`, `starlette`, `mcp` are in `[project.optional-dependencies] runtime`, not core deps. You can use Brain/Heart/Cognitive as a library without installing the server.

## Error Handling

- All REST handlers wrapped in try/except → return JSONResponse with error details
- MCP tool calls wrapped in try/except → return TextContent with error message
- Runner API errors → create TurnResult with error, still call post_turn
- Component initialization failure in main.py → log error, exit(1)
- DB connectivity failure in /health → return {"status": "unhealthy"}

## Migration Notes

- **docker-compose.yml is replaced** — new version adds `nous` service alongside `postgres`
- **pyproject.toml updated** — new `runtime` optional dependency group
- **.env.example updated** — new env vars for Anthropic API key, model, port
- **No schema changes** — all tables exist from 001-postgres-scaffold
- **New `nous/api/` directory** — 3 new files (runner.py, rest.py, mcp.py)
- **New `nous/main.py`** — entry point
- **New `Dockerfile`** — builds the container
