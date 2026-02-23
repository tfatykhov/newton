# 005-Runtime Implementation Plan

**Spec:** [005-runtime.md](005-runtime.md)
**Review Decision:** `fd10b475` (3-agent review team)
**Plan Decision:** `202ced82` (synthesis)
**Reviewers:** nous-arch-005, nous-api-005, nous-devil-005

## Review Summary

3-agent review found **5 P0 blockers**, **11 P1 important**, **10 P2 nice-to-have** issues.
High convergence on top 3: Database method names, event loop lifecycle, Dockerfile install order.

### Consolidated Fixes (deduplicated across all 3 reviewers)

| ID | Severity | Issue | Fix | Flagged By |
|----|----------|-------|-----|------------|
| F1 | P0 | `Database.initialize()`/`close()` don't exist | Use `connect()`/`disconnect()` | all 3 |
| F2 | P0 | Event loop lifecycle broken (3 loops) | Starlette lifespan context manager | all 3 |
| F3 | P0 | `atexit` shutdown on wrong loop | Same as F2 — lifespan handles both | all 3 |
| F4 | P0 | Shared EmbeddingProvider double-close | `Heart(..., owns_embeddings=False)` | arch + devil |
| F5 | P0 | `Dockerfile -e` install before source copy | Non-editable install, reorder copies | api + arch |
| F6 | P0 | `identity_prompt` ordering dependency | Config changes MUST be Phase A | devil |
| F7 | P1 | Empty `Brain.query("")` returns nothing for `/decisions` | Add `Brain.list_decisions()` method | api + devil |
| F8 | P1 | Empty `episodes.search("")` returns nothing for `/episodes` | Use `Heart.list_episodes()` (exists) | api + devil |
| F9 | P1 | `CognitiveLayer._frames` accessed from REST | Add public `CognitiveLayer.list_frames()` | all 3 |
| F10 | P1 | `end_session()` missing `agent_id`, `session_id` args | Correct call: `end_session(agent_id, session_id, reflection=...)` | api |
| F11 | P1 | `nous_teach` ProcedureInput missing `name` mapping | Derive `name` from first 100 chars of `content` | api + devil |
| F12 | P1 | MCP mount pattern needs verification | Test `StreamableHTTPSessionManager` ASGI compatibility | all 3 |
| F13 | P1 | `NOUS_PORT` dual-meaning confusion | Use `validation_alias="NOUS_HOST_PORT"` for docker, keep `port` for app | arch + devil |
| F14 | P1 | GET /censors wrong method | Use `Heart.list_censors()` not `censors.search()` | api + devil |
| F15 | P1 | Empty ANTHROPIC_API_KEY not caught at startup | Log warning at startup, fail-fast on first /chat | devil |
| F16 | P1 | `nous_recall` merge strategy unspecified | Normalize to common schema with source field | devil |
| F17 | P1 | In-memory conversation leak | MAX_CONVERSATIONS=100 with LRU eviction | devil |
| F18 | P1 | `model` field name ambiguous | Rename to `llm_model` with `validation_alias="NOUS_MODEL"` | arch |
| F19 | P2 | Concurrent same-session race condition | Defer to v0.2.0, document as known limitation | devil |
| F20 | P2 | F004 → 005 drift undocumented | Add "Deferred from F004" section to spec | devil |
| F21 | P2 | `nous_decide` stakes param discarded | Pass through to pre_turn or document as unused | devil |
| F22 | P2 | REST error status codes underspecified | Implement 400/404/500 explicitly | api |
| F23 | P2 | No count methods for /status | Write raw SQL COUNT queries in rest.py | api |
| F24 | P2 | Reflection prompt hardcoded | Make class constant on AgentRunner | arch |
| F25 | P2 | MCP session manager cleanup on shutdown | Add to lifespan shutdown | devil |

### Lead Overrides

- **F19 (concurrent race):** Devil flagged as P1, lead downgrades to P2. asyncio.Lock per session is over-engineering for v0.1.0 single-instance. Document as known limitation.
- **F13 (NOUS_PORT):** Simplify — remove `NOUS_PORT` from Settings entirely. In docker-compose, the container always binds 8000 internally. `NOUS_PORT` only controls the HOST mapping. Settings.port is always 8000 unless explicitly overridden by `NOUS_PORT` env var (which is fine).
- **F18 (model name):** Keep as `model` for now. The NOUS_ prefix already disambiguates (`NOUS_MODEL`). Renaming adds churn.
- **F21 (nous_decide stakes):** Include — pass stakes in the user message prefix: "Decision (stakes: high): Should we..."

---

## Implementation Phases

### Phase A: Pre-work (config + helper methods)

**Files modified:**
- `nous/config.py` — Add runtime fields
- `nous/brain/brain.py` — Add `list_decisions()` method
- `nous/cognitive/layer.py` — Add public `list_frames()` method

**Config additions** (to existing `Settings` class):
```python
# Runtime
host: str = "0.0.0.0"
port: int = 8000
anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")

# Agent identity
agent_name: str = "Nous"
agent_description: str = "A thinking agent that learns from experience"
identity_prompt: str = ""

# MCP
mcp_enabled: bool = True

# LLM
model: str = "claude-sonnet-4-5-20250514"
max_tokens: int = 4096
```

**Brain.list_decisions():**
```python
async def list_decisions(
    self,
    limit: int = 20,
    offset: int = 0,
    agent_id: str | None = None,
    session: AsyncSession | None = None,
) -> tuple[list[DecisionSummary], int]:
    """List decisions ordered by created_at DESC. Returns (decisions, total_count)."""
    # Raw SQL: SELECT from brain.decisions ORDER BY created_at DESC LIMIT/OFFSET
    # Also SELECT COUNT(*) for total
```

**CognitiveLayer.list_frames():**
```python
async def list_frames(self, agent_id: str, session: AsyncSession | None = None) -> list:
    """Public delegation to FrameEngine.list_frames()."""
    return await self._frames.list_frames(agent_id, session=session)
```

### Phase B: Agent Runner (`nous/api/runner.py`)

**New file ~180 lines.**

Key corrections from spec:
- `Message` and `Conversation` dataclasses as specified
- `AgentRunner.__init__` takes `cognitive: CognitiveLayer` and `settings: Settings`
- `run_turn()` calls:
  - `self._cognitive.pre_turn(agent_id, session_id, user_message)` — note arg order
  - httpx POST to `https://api.anthropic.com/v1/messages`
  - `self._cognitive.post_turn(agent_id, session_id, turn_result, turn_context)`
- `end_conversation()` calls `self._cognitive.end_session(agent_id, session_id, reflection=...)`
- `_format_messages()` caps at last 20 messages
- MAX_CONVERSATIONS = 100, LRU eviction via OrderedDict (F17)
- Reflection prompt as class constant `REFLECTION_PROMPT` (F24)
- On API error: log, create `TurnResult(response_text="", error=str(e))`, still call post_turn

### Phase C: REST API (`nous/api/rest.py`)

**New file ~280 lines.**

Key corrections from spec:
- `create_app()` takes all components as args
- `/chat` POST: call `runner.run_turn()`, return response + session_id + frame + decision_id
- `/chat/{session_id}` DELETE: call `runner.end_conversation()`
- `/status` GET: `brain.get_calibration()` + raw SQL COUNT queries for memory stats (F23)
- `/decisions` GET: `brain.list_decisions(limit, offset)` — NOT `brain.query("")` (F7)
- `/decisions/{id}` GET: `brain.get(id)` — return 404 if not found (F22)
- `/episodes` GET: `heart.list_episodes(limit)` — NOT `episodes.search("")` (F8)
- `/facts` GET: `heart.search_facts(q, limit)` — require `q` param, return 400 if missing (F22)
- `/censors` GET: `heart.list_censors()` — NOT `censors.search()` (F14)
- `/frames` GET: `cognitive.list_frames(agent_id)` — NOT `cognitive._frames` (F9)
- `/calibration` GET: `brain.get_calibration()`
- `/health` GET: `SELECT 1` via database session
- Error handling: explicit 400/404/500 status codes (F22)

### Phase D: MCP Server (`nous/api/mcp.py`)

**New file ~200 lines.**

Key corrections from spec:
- 5 tools: `nous_chat`, `nous_recall`, `nous_status`, `nous_teach`, `nous_decide`
- `nous_recall`: normalize Brain.query() DecisionSummary + Heart.recall() RecallResult into common response format with `source` field (F16)
- `nous_teach` procedure path: derive `name` from `content[:100]` (F11)
- `nous_decide`: include stakes in message prefix, don't just discard it (F21)
- MCP mount: verify `StreamableHTTPSessionManager` ASGI pattern against mcp library docs (F12). If `handle_request` isn't directly ASGI-compatible, use the library's Starlette adapter.
- Wrap all tool handlers in try/except, return `TextContent` with error message

### Phase E: Entry Point (`nous/main.py`)

**New file ~100 lines.**

Key corrections from spec — this is the most changed file:
- **Use Starlette lifespan** instead of manual event loop + atexit (F2, F3):
  ```python
  @asynccontextmanager
  async def lifespan(app):
      # Startup
      components = await create_components(settings)
      app.state.update(components)
      yield
      # Shutdown (reverse order)
      await shutdown(components)
  ```
- `create_components()`: use `database.connect()` not `initialize()` (F1)
- `create_components()`: use `Heart(..., owns_embeddings=False)` (F4)
- `shutdown()`: use `database.disconnect()` not `close()` (F1)
- Log warning if `anthropic_api_key` is empty (F15)
- `build_app()`: pass lifespan to Starlette constructor
- MCP mount at `/mcp` if `settings.mcp_enabled`
- `main()`: just `Settings()` + logging + `uvicorn.run(app, ...)`

### Phase F: Docker & Config Files

**Modified files:**
- `pyproject.toml` — Add `[project.optional-dependencies] runtime` group
- `docker-compose.yml` — Add `nous` service alongside existing `postgres`
- `Dockerfile` — New file, non-editable install (F5)
- `.env.example` — Updated with new env vars
- `nous/api/__init__.py` — Update docstring (file already exists)

**Dockerfile fix** (F5):
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
COPY nous/ nous/
RUN pip install --no-cache-dir ".[runtime]"
COPY sql/ sql/
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["python", "-m", "nous.main"]
```

### Phase G: Tests

**New files:**
- `tests/test_runner.py` (~200 lines)
- `tests/test_rest.py` (~250 lines)
- `tests/test_mcp.py` (~180 lines)

**Test infrastructure:**
- `MockCognitiveLayer`: returns preset `TurnContext` from `pre_turn()`, records `post_turn()` / `end_session()` calls
- `MockAgentRunner`: returns canned responses from `run_turn()`, tracks call history
- httpx `MockTransport` for Anthropic API simulation
- Starlette `TestClient` (sync, uses httpx under the hood) for REST tests
- MCP tool handlers tested directly (call the `call_tool` function)
- Real Postgres fixtures from existing `conftest.py` for endpoints that hit DB

---

## Implementation Team Structure

Same proven pattern as 004:

| Agent | Role | Phases | Isolation |
|-------|------|--------|-----------|
| python-engineer-005 | Implementer | A → B → C → D → E → F | worktree |
| test-engineer-005 | Tester | G (parallel after Phase B) | worktree |
| code-reviewer-005 | Reviewer | Review after all phases | reads main worktree |

**Pipeline:**
1. python-engineer-005 implements Phases A-F sequentially
2. test-engineer-005 starts writing tests after Phase B is done (runner.py exists)
3. code-reviewer-005 reviews all code after implementation complete
4. Lead runs full test suite, applies fixes, merges

---

## Deferred to v0.2.0

Per devil's advocate review (F004 → 005 drift):
- WebSocket `/ws` endpoint (F004)
- Dashboard service (F004)
- `config/agent.yaml` file-based config (F004)
- Concurrent session locking (F19)
- Request body size limits
- CORS middleware
- Authentication (API keys / JWT)
- Persistent conversation state (DB-backed)
- HITL approval gates for memory writes (D10)

---

## Acceptance Criteria

1. `docker compose up -d` starts both postgres and nous containers
2. `curl POST /chat` returns LLM response with session tracking
3. `curl DELETE /chat/{session}` triggers end_session with reflection
4. `curl GET /status` returns agent info + calibration + memory counts
5. `curl GET /health` returns healthy when DB is up, unhealthy when down
6. All 11 REST endpoints return correct responses
7. MCP tools work when connected via mcp client
8. Graceful shutdown closes all resources in correct order
9. All tests pass: `pytest tests/test_runner.py tests/test_rest.py tests/test_mcp.py -v`
