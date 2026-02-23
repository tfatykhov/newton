# 005-Runtime Implementation Plan (SDK Edition)

**Spec:** [005-runtime.md](005-runtime.md) (SDK rewrite, commit `b6898f6`)
**Previous Plan:** `202ced82` (Round 1 — httpx-based)
**This Plan Decision:** `7b214c00` (Round 2 — SDK migration)
**Review Team:** nous-arch (`db0239e6`), nous-api (`d022b56f`), nous-devil (`03790f0d`)

## Executive Summary

The updated 005-runtime spec rewrites the agent runner for **Claude Agent SDK** integration, replacing direct httpx calls with SDK-managed tool use. This transforms Nous from "context-only RAG" to "tool-augmented LLM" — Claude can call `record_decision`, `learn_fact`, `recall_deep`, and `create_censor` during a turn.

**Critical finding:** The spec contains **regressions of already-fixed bugs** from Round 1 AND **incorrect SDK API assumptions**. Strategy: **MERGE not REPLACE** — keep existing fixes, layer SDK changes on top.

---

## Review Summary

3-agent review found **10 P0 blockers**, **9 P1 important**, **5 P2 nice-to-have** issues (deduplicated across all 3 reviewers).

### P0 Blockers (10) — Must fix before/during implementation

| ID | Issue | Fix | Flagged By |
|----|-------|-----|------------|
| F1 | `Database.initialize()`/`close()` still wrong in spec | Already fixed in codebase: `connect()`/`disconnect()` — **do not regress** | all 3 |
| F2 | Event loop lifecycle broken (spec uses manual loop + atexit) | Already fixed: Starlette lifespan — **do not regress** | all 3 |
| F3 | `atexit` + `asyncio.run()` creates wrong loop | Same as F2 — lifespan handles both — **do not regress** | all 3 |
| F4 | SDK `query()` doesn't support custom tools | Must use `ClaudeSDKClient` instead of `query()` | devil |
| F5 | SDK `query()` API signature wrong | Spec passes `message`/`messages` params that don't exist | devil |
| F6 | SDK `query()` yields wrong event types | Actual SDK yields `Message` objects, not `event.type == "text"` | devil |
| F7 | `ClaudeAgentOptions` field mismatches | `mcp_servers` expects dict not list; `permission_mode` values wrong; no `max_tokens` field | devil |
| F8 | Dockerfile editable install regression | Already fixed: non-editable `pip install ".[runtime]"` — **do not regress** | api + arch |
| F9 | `pyproject.toml` removes pgvector + httpx | Keep existing deps — pgvector needed for DB, httpx for tests | api |
| F10 | `@tool` decorator + closure pattern conflated | Module-level `@tool` stubs raise NotImplementedError and are never used. Use closures directly. | devil |

### P1 Important (9) — Fix during implementation

| ID | Issue | Fix | Flagged By |
|----|-------|-----|------------|
| F11 | `CalibrationReport.reviewed_count` wrong field name | Actual: `reviewed_decisions` — already correct in codebase | devil + api |
| F12 | `brain.query("")` for listing decisions | Already fixed: `brain.list_decisions()` — **do not regress** | api + devil |
| F13 | `rest.py` missing lifespan parameter | Already fixed in codebase — **do not regress** | api |
| F14 | `cognitive._frames` private attribute access | Already fixed: public `cognitive.list_frames()` — **do not regress** | all 3 |
| F15 | MCP uses nonexistent `StreamableHTTPSessionManager` | Already fixed: `StreamableHTTPServerTransport` + `MCPTransportManager` — **do not regress** | api + arch |
| F16 | SDK `allowed_tools` naming convention wrong | Must use `mcp__nous__record_decision` prefix format + capitalize builtins (`Bash`, `Read`) | devil |
| F17 | `heart.recall()` doesn't search Brain decisions | `recall_deep` tool for `memory_type="all"` needs both `heart.recall()` AND `brain.query()` | devil |
| F18 | `nous_teach` hardcodes `category="rule"` | Already fixed: uses `category=domain` from args — **do not regress** | api |
| F19 | Spec drops LRU conversation eviction | Keep existing `OrderedDict` + `MAX_CONVERSATIONS=100` from codebase | arch + api |

### P2 Nice-to-have (5)

| ID | Issue | Fix |
|----|-------|-----|
| F20 | `FactSummary.source` doesn't exist | Check schema, add field or remove access |
| F21 | Known test fixture issues (SAVEPOINT, guardrail format) | Fix in conftest.py during test phase |
| F22 | No SDK workspace directory in Dockerfile | Add `mkdir -p /tmp/nous-workspace` |
| F23 | `nous_decide` format difference | Keep existing `"Decision (stakes: {stakes}): {question}"` format (richer) |
| F24 | Missing config fields for dual auth + SDK | Additive changes to Settings class — no conflicts |

### Lead Overrides

- **F4-F7 (SDK API mismatches):** Devil found `query()` doesn't support custom tools. **Confirmed via SDK docs.** The spec's entire runner.py must be redesigned around `ClaudeSDKClient` instead of `query()`. This is the largest change from the spec.
- **F10 (@tool decorator):** Remove module-level `@tool` stubs entirely. The closure pattern in `create_nous_tools()` is the correct approach — register closures with the SDK's MCP server directly.
- **F16 (tool naming):** Need to verify actual SDK tool name resolution. May require `mcp__` prefix or may be auto-prefixed by SDK. Verify during Phase 0.
- **F17 (recall_deep scope):** Accept for v0.1.0. `heart.recall()` covers episodes, facts, procedures. Adding brain.query() to the "all" path is a simple enhancement but needs merge/dedup logic.

---

## Strategy: MERGE not REPLACE

The spec should be treated as a **design reference for new SDK functionality**, NOT as a replacement for existing working code. Files with already-applied fixes must NOT be overwritten with spec code.

| File | Action | Rationale |
|------|--------|-----------|
| `nous/config.py` | **MODIFY** (add 6 fields) | Additive. No conflicts. |
| `nous/api/tools.py` | **CREATE** (new, ~250 lines) | Entirely new file. Tool closures from spec are sound. SDK registration needs adaptation. |
| `nous/api/runner.py` | **REWRITE** (keep good patterns) | Core change: httpx → SDK. Keep: OrderedDict LRU, named constants, REFLECTION_PROMPT. Add: frame instructions, safety net. |
| `nous/api/rest.py` | **NO CHANGE** | Existing is superior. Spec regresses pagination, field names, error handling, lifespan. |
| `nous/api/mcp.py` | **NO CHANGE** | Existing is correct. Spec regresses to nonexistent StreamableHTTPSessionManager. |
| `nous/main.py` | **MODIFY** (small) | Keep lifespan, connect/disconnect, LazyProxy. Only change: runner constructor adds brain+heart, SDK logging. |
| `Dockerfile` | **MODIFY** (add workspace) | Keep non-editable install. Add SDK workspace mkdir. |
| `docker-compose.yml` | **MODIFY** (add env vars) | Add SDK env vars (max_turns, permission_mode, workspace, auth_token). |
| `pyproject.toml` | **MODIFY** (add dep) | Add `claude-agent-sdk` to runtime deps. Keep pgvector, httpx. |
| `.env.example` | **MODIFY** (add vars) | Add SDK settings + auth_token. |
| `tests/test_runner.py` | **REWRITE** | Mock SDK instead of httpx. New tests for tool calls, frame instructions, safety net. |
| `tests/test_tools.py` | **CREATE** (new, ~180 lines) | Test 4 tool closures directly against real Brain/Heart. |
| `tests/test_rest.py` | **NO CHANGE** | MockAgentRunner is SDK-agnostic. All 16 tests work as-is. |
| `tests/test_mcp.py` | **NO CHANGE** | MCPTransportManager extraction works. All 8 tests work as-is. |

---

## Implementation Phases

### Phase 0: SDK Verification (GATE — must pass before proceeding)

**Goal:** Verify `claude-agent-sdk` exists, confirm actual API surface.

1. `pip install claude-agent-sdk` — does it install?
2. Inspect actual imports: `ClaudeSDKClient`, `ClaudeAgentOptions`, `query`, MCP integration
3. Confirm:
   - Does `ClaudeSDKClient` support custom MCP tools? (devil says yes, query doesn't)
   - What is the actual `ClaudeAgentOptions` signature?
   - How does `mcp_servers` work — dict, list, or something else?
   - What does the response/event stream look like?
   - Tool name format: bare names, `mcp__` prefixed, or configurable?
4. Document findings in this plan (update this section)

**If SDK doesn't exist or API is fundamentally different:** Fall back to raw Anthropic Messages API with tool_use (native tool calling without SDK). This is a proven pattern and doesn't require a third-party SDK.

**Fallback architecture (if SDK unavailable):**
```
runner._call_llm() sends tools=[...] in Messages API request
→ response may contain tool_use content blocks
→ runner executes tool calls in-process (same closure pattern)
→ runner sends tool_result back to API
→ loop until Claude produces text-only response
```

### Phase A: Config + Dependencies

**Files modified:**
- `nous/config.py` — Add 6 new settings fields
- `pyproject.toml` — Add `claude-agent-sdk` (or alternative) to runtime deps
- `docker-compose.yml` — Add SDK env vars
- `.env.example` — Add SDK vars + auth_token
- `Dockerfile` — Add workspace mkdir

**Config additions** (to existing `Settings` class):
```python
# Dual auth (add alongside existing anthropic_api_key)
anthropic_auth_token: str = Field("", validation_alias="ANTHROPIC_AUTH_TOKEN")

# SDK settings
sdk_max_turns: int = 10        # Max tool use iterations per query
sdk_permission_mode: str = "auto"
sdk_workspace: str = "/tmp/nous-workspace"
sdk_allowed_tools: list[str] = Field(
    default_factory=lambda: [
        "record_decision", "learn_fact", "recall_deep", "create_censor",
    ]
)
```

### Phase B: In-Process Tools (`nous/api/tools.py`)

**New file ~250 lines.**

This is the core new capability — 4 tools for Claude to use during turns:

1. **`record_decision`** — Write a decision to Brain via `brain.record(RecordInput(...))`
2. **`learn_fact`** — Store a fact to Heart via `heart.learn(FactInput(...))`
3. **`recall_deep`** — Search all memory types (heart.recall + brain.query for decisions)
4. **`create_censor`** — Create a guardrail via `heart.add_censor(CensorInput(...))`

**Architecture:**
- `create_nous_tools(brain, heart)` → returns `dict[str, callable]` of async closures
- Each closure gets Brain/Heart via closure context (zero latency, in-process)
- Each closure returns MCP-compliant `{"content": [{"type": "text", "text": "..."}]}`
- All wrapped in try/except → return error message as tool result
- **No module-level `@tool` stubs** — remove the conflated pattern (F10)

**SDK registration (depends on Phase 0 findings):**
- If SDK: `create_nous_mcp_server(brain, heart)` returns server for `mcp_servers` config
- If fallback: `create_anthropic_tools(brain, heart)` returns tool definitions for Messages API `tools` param

### Phase C: Agent Runner Rewrite (`nous/api/runner.py`)

**Full rewrite ~220 lines.** This is the critical file.

**Keep from existing:**
- `Message` and `Conversation` dataclasses
- `OrderedDict` with LRU eviction, `MAX_CONVERSATIONS=100` (F19)
- `MAX_HISTORY_MESSAGES = 20` named constant
- `REFLECTION_PROMPT` class constant
- `_format_messages()` and `_get_or_create_conversation()`
- Error handling: always call post_turn even on error

**Change:**
- Constructor: `AgentRunner(cognitive, brain, heart, settings)` — adds brain+heart for tools
- `start()`: Set up SDK auth env vars + create in-process MCP server
- `close()`: SDK cleanup (if needed)
- `_call_llm()` → replaced by SDK client query with tool loop
- `run_turn()`: SDK query with system prompt + tools + conversation history, process Message stream
- New `_add_frame_instructions(turn_context)`: Adds "You MUST call record_decision" etc. based on frame
- New safety net: After post_turn, check if decision frame was active but `record_decision` wasn't called → log warning

**SDK integration (depends on Phase 0):**
```python
# If ClaudeSDKClient:
async with ClaudeSDKClient(options) as client:
    response = await client.query(prompt, messages=history)
    # Process response.content blocks

# If Messages API fallback:
while True:
    response = await self._client.post("/v1/messages", json={...tools...})
    if no tool_use blocks: break
    execute tools, append tool_results, continue
```

### Phase D: Main Entry Point (`nous/main.py`)

**MINIMAL changes to existing file.**

1. Change `AgentRunner(cognitive, settings)` → `AgentRunner(cognitive, brain, heart, settings)`
2. Add SDK startup logging: `settings.sdk_max_turns`, `settings.sdk_workspace`
3. Update anthropic_api_key warning to also check `anthropic_auth_token`
4. **DO NOT** change lifespan, connect/disconnect, LazyProxy, or shutdown order

### Phase E: Tests

| Test File | Action | Lines | Strategy |
|-----------|--------|-------|----------|
| `tests/test_tools.py` | **CREATE** | ~180 | Test 4 tool closures against real Brain/Heart. 12 tests. |
| `tests/test_runner.py` | **REWRITE** | ~220 | Mock SDK/API instead of httpx. New: tool_calls, frame_instructions, safety_net. Keep: all existing test scenarios. |
| `tests/test_rest.py` | **NO CHANGE** | 0 | MockAgentRunner is SDK-agnostic. |
| `tests/test_mcp.py` | **NO CHANGE** | 0 | MCPTransportManager works regardless of SDK. |

**test_tools.py tests:**
```
test_create_nous_tools — returns dict with 4 functions
test_record_decision_success — calls brain.record, returns success
test_record_decision_validation_error — invalid confidence → error message
test_record_decision_brain_error — brain.record raises → error message
test_learn_fact_success — calls heart.learn, returns success
test_learn_fact_duplicate — deduplication → still success
test_recall_deep_all — searches all memory types
test_recall_deep_decisions — brain.query only
test_recall_deep_facts — heart.search_facts only
test_recall_deep_empty — no results → "No results found."
test_create_censor_success — calls heart.add_censor, returns success
test_create_censor_invalid_action — invalid enum → error message
```

**test_runner.py rewrite (mock SDK, not httpx):**
```
test_run_turn_basic — mock SDK returns text → response returned
test_run_turn_creates_conversation — new session creates Conversation
test_run_turn_appends_messages — history grows each turn
test_run_turn_calls_pre_turn — cognitive.pre_turn called
test_run_turn_calls_post_turn — cognitive.post_turn called
test_run_turn_with_tool_calls — SDK returns tool results → post_turn sees them
test_run_turn_sdk_error — SDK raises → error returned, post_turn still called
test_run_turn_history_capped — 20 message limit
test_run_turn_frame_instructions — decision frame → "MUST call record_decision" in prompt
test_run_turn_safety_net — decision frame + no record_decision → warning logged
test_end_conversation — removes from dict, calls end_session
test_end_conversation_with_reflection — >= 3 turns → reflection generated
test_end_conversation_nonexistent — no error
test_start_with_api_key — env var set correctly
test_start_with_auth_token — auth_token takes precedence
test_start_no_credentials — RuntimeError raised
test_lru_eviction — 101st conversation evicts oldest
```

---

## Implementation Team Structure

| Agent | Role | Phases | Isolation |
|-------|------|--------|-----------|
| sdk-engineer | Implementer | 0 → A → B → C → D | worktree |
| test-engineer | Tester | E (starts after Phase B) | worktree |
| code-reviewer | Reviewer | After all phases complete | reads worktree |

**Pipeline:**
1. sdk-engineer runs Phase 0 (SDK verification gate)
2. If gate passes: A → B → C → D sequentially
3. test-engineer starts after Phase B (tools.py exists, runner.py exists)
4. code-reviewer reviews all code
5. Lead runs full test suite, applies fixes, merges

---

## Architectural Decision: SDK vs Fallback

The spec assumes `claude-agent-sdk` with `query()`. Devil's advocate proved `query()` doesn't support custom tools. Two options:

### Option A: ClaudeSDKClient (if SDK available)
- `ClaudeSDKClient` supports custom tools via `mcp_servers` config
- Manages agentic tool loop internally
- We manage conversation history, SDK manages tool execution
- **Pro:** Less code, SDK handles tool orchestration
- **Con:** SDK is a dependency we don't control, API may change

### Option B: Raw Messages API with tool_use (fallback)
- Standard Anthropic Messages API with `tools` parameter
- We implement the tool loop ourselves (send tools → get tool_use → execute → send tool_result → repeat)
- **Pro:** No SDK dependency, we control everything, well-documented API
- **Con:** More code (~50 lines for tool loop), manual tool registration

**Recommendation:** Try Option A first (Phase 0 gate). If SDK unavailable or API too different, fall back to Option B. The tool closures in `tools.py` work identically for both — only the registration and execution loop differ.

---

## Deferred to v0.2.0

Carried from Round 1 + new items:
- WebSocket `/ws` endpoint
- Dashboard service
- `config/agent.yaml` file-based config
- Concurrent session locking
- Request body size limits / CORS middleware
- Authentication (API keys / JWT)
- Persistent conversation state (DB-backed)
- HITL approval gates for memory writes
- Auto-record safety net (currently just logs warning)
- `ClaudeSDKClient` multi-turn sessions (P1 from spec's future enhancements)
- HookMatcher for pre/post tool use hooks

---

## Acceptance Criteria

1. `docker compose up -d` starts both postgres and nous containers
2. `POST /chat` returns LLM response with session tracking
3. **Claude can call `record_decision` during a turn** (the core new capability)
4. **Claude can call `learn_fact`, `recall_deep`, `create_censor` during a turn**
5. Frame-conditional instructions appear in system prompt ("You MUST call record_decision")
6. Safety net detects missed tool calls and logs warning
7. `DELETE /chat/{session}` triggers end_session with reflection (>= 3 turns)
8. `GET /status` returns agent info + calibration + memory counts
9. `GET /health` returns healthy/unhealthy
10. All 11 REST endpoints return correct responses (unchanged from Round 1)
11. External MCP tools work (unchanged from Round 1)
12. Graceful shutdown via Starlette lifespan (unchanged from Round 1)
13. All tests pass: `pytest tests/test_runner.py tests/test_rest.py tests/test_mcp.py tests/test_tools.py -v`
