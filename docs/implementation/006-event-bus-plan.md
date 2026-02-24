# 006 Event Bus — Implementation Plan

**Status:** READY FOR IMPLEMENTATION
**Branch:** `feat/006-event-bus`
**Spec:** `docs/implementation/006-event-bus.md`
**Review:** 3-agent analysis team (nous-arch-006, nous-integration-006, nous-devil-006)
**Decision IDs:** f9740aea (team), 6e00da4d (arch), 416be426 (integration), 4189ef90 (devil), 48694bba (plan)

## Review Summary

| Reviewer | P0 | P1 | P2 | Key Finding |
|----------|----|----|----|----|
| nous-arch-006 | 5 | 3 | 5 | Sleep handler blocks bus; timeout pipeline broken end-to-end |
| nous-integration-006 | 3 | 5 | 5 | bus=None drops events; schema/ORM gaps |
| nous-devil-006 | 10 | 5 | 8 | Every spec snippet vs codebase mismatch cataloged |

**High convergence (all 3 found independently):**
1. Brain.emit_event() signature mismatch — no agent_id/session_id params
2. CognitiveLayer.__init__ drops identity_prompt when adding bus
3. Timeout pipeline broken — no episode_id/transcript in timeout events
4. FactExtractor dedup: wrong attribute (.similarity vs .score) + unreachable threshold

## Consolidated Fixes (13 P0, 8 P1)

### P0 Fixes (must apply)

| # | Issue | Fix |
|---|-------|-----|
| P0-1 | Brain.emit_event() has no agent_id/session_id params | Remove kwargs from adapter; put session_id inside data dict |
| P0-2 | CognitiveLayer.__init__ spec drops identity_prompt | Signature: `__init__(self, brain, heart, settings, identity_prompt="", *, bus=None)` |
| P0-3 | main.py drops identity_prompt when adding bus | Call: `CognitiveLayer(brain, heart, settings, settings.identity_prompt, bus=bus)` |
| P0-4 | Heart.update_episode_summary() doesn't exist | Add to EpisodeManager + Heart delegation |
| P0-5 | Episode ORM missing structured_summary column | Add `mapped_column(JSONB, nullable=True)` |
| P0-6 | Episode ORM missing user_id, user_display_name | Add 2 `mapped_column(String(100), nullable=True)` |
| P0-7 | FactExtractor dedup uses .similarity (doesn't exist) | Use `.score`, lower threshold to 0.65 for hybrid search |
| P0-8 | 8 config fields missing from Settings | Add all fields per Phase 2 below |
| P0-9 | SessionMetadata missing transcript field | Add `transcript: list[str] = field(default_factory=list)` |
| P0-10 | Timeout session_ended has no episode_id/transcript | Monitor calls cognitive.end_session() instead of emitting raw event |
| P0-11 | Sleep handler blocks bus (sequential dispatch) | Handler spawns asyncio.Task, returns immediately |
| P0-12 | EpisodeInput missing user_id/user_display_name | Add optional fields to schema |
| P0-13 | _safe_handle catches Exception, not BaseException | CancelledError can escape; catch BaseException, re-raise CancelledError |

### P1 Fixes (should apply)

| # | Issue | Fix |
|---|-------|-----|
| P1-1 | bus=None backward compat drops events | Add else branch: `await self._brain.emit_event(...)` |
| P1-2 | stop() drain races with _process_loop | Set _running=False → await _task → THEN drain |
| P1-3 | Double session_ended (timeout + explicit) | Monitor listens for session_ended to clean tracking; cognitive.end_session removes from monitor |
| P1-4 | _last_activity dict leaks for explicit-end sessions | Monitor cleans on session_ended event |
| P1-5 | runner.py has no user_id params | Add explicit user_id/user_display_name params to run_turn/stream_chat |
| P1-6 | init.sql needs CREATE TABLE updates, not just ALTER | Update init.sql CREATE TABLE + provide migration SQL |
| P1-7 | EpisodeDetail/EpisodeSummary missing new fields | Add structured_summary, user_id, user_display_name |
| P1-8 | FactExtractor drops category from LLM response | Add `category=fact.get("category")` to FactInput |

---

## Implementation Phases

### Phase 0: Pre-Work (config, schemas, ORM, SQL)

**Files:** `nous/config.py`, `nous/cognitive/schemas.py`, `nous/heart/schemas.py`, `nous/storage/models.py`, `sql/init.sql`

#### 0A: Config — Add 8 new Settings fields

```python
# config.py — add to Settings class
event_bus_enabled: bool = True
episode_summary_enabled: bool = True
fact_extraction_enabled: bool = True
sleep_enabled: bool = True
background_model: str = Field(
    default="claude-sonnet-4-5-20250514",
    validation_alias="NOUS_BACKGROUND_MODEL",
)
session_idle_timeout: int = Field(
    default=1800,
    validation_alias="NOUS_SESSION_TIMEOUT",
)
sleep_timeout: int = Field(
    default=7200,
    validation_alias="NOUS_SLEEP_TIMEOUT",
)
sleep_check_interval: int = Field(
    default=60,
    validation_alias="NOUS_SLEEP_CHECK_INTERVAL",
)
```

#### 0B: SessionMetadata — Add transcript field

```python
# cognitive/schemas.py — add to SessionMetadata dataclass
transcript: list[str] = field(default_factory=list)
```

#### 0C: Heart Schemas — Add fields to EpisodeInput, EpisodeDetail, EpisodeSummary

```python
# heart/schemas.py — EpisodeInput additions
user_id: str | None = None
user_display_name: str | None = None

# heart/schemas.py — EpisodeDetail additions
structured_summary: dict | None = None
user_id: str | None = None
user_display_name: str | None = None

# heart/schemas.py — EpisodeSummary additions
structured_summary: dict | None = None
```

#### 0D: Episode ORM Model — Add 3 columns

```python
# storage/models.py — Episode class additions
structured_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
user_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
user_display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
```

#### 0E: SQL — Update init.sql CREATE TABLE + migration

In `sql/init.sql`, add to `heart.episodes` CREATE TABLE:
```sql
structured_summary JSONB,
user_id VARCHAR(100),
user_display_name VARCHAR(100),
```

Add indexes after CREATE TABLE:
```sql
CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON heart.episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_summary_gin ON heart.episodes USING GIN(structured_summary);
```

Create `sql/migrations/006_event_bus.sql` for existing installs:
```sql
-- 006: Event Bus schema additions
ALTER TABLE heart.episodes ADD COLUMN IF NOT EXISTS structured_summary JSONB;
ALTER TABLE heart.episodes ADD COLUMN IF NOT EXISTS user_id VARCHAR(100);
ALTER TABLE heart.episodes ADD COLUMN IF NOT EXISTS user_display_name VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON heart.episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_summary_gin ON heart.episodes USING GIN(structured_summary);
```

#### 0F: Heart — Add update_episode_summary method chain

```python
# heart/episodes.py — add to EpisodeManager
async def update_summary(
    self, episode_id: UUID, summary: dict, session: AsyncSession | None = None
) -> None:
    """Store structured summary on episode."""
    if session is None:
        async with self.db.session() as session:
            await self._update_summary(episode_id, summary, session)
            await session.commit()
            return
    await self._update_summary(episode_id, summary, session)

async def _update_summary(
    self, episode_id: UUID, summary: dict, session: AsyncSession
) -> None:
    stmt = select(Episode).where(Episode.id == episode_id)
    result = await session.execute(stmt)
    episode = result.scalar_one_or_none()
    if episode:
        episode.structured_summary = summary
        await session.flush()
```

```python
# heart/heart.py — add delegation method
async def update_episode_summary(self, episode_id: UUID, summary: dict, session=None):
    """Store structured summary on episode."""
    await self.episodes.update_summary(episode_id, summary, session=session)
```

Also update `_to_detail()` in `episodes.py` to map new columns:
```python
structured_summary=episode.structured_summary,
user_id=episode.user_id,
user_display_name=episode.user_display_name,
```

And update `EpisodeManager.start()` to accept and store user_id/user_display_name from EpisodeInput.

---

### Phase 1: EventBus Core (~120 lines)

**File:** `nous/events.py` (NEW)

Use spec Phase A code with these fixes:

#### Fix 1: _safe_handle — catch BaseException
```python
async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
    """Run handler with error isolation. Never propagates (except CancelledError)."""
    try:
        await handler(event)
    except asyncio.CancelledError:
        raise  # Propagate cancellation
    except BaseException:
        logger.exception(
            "Handler %s failed for event %s",
            handler.__qualname__,
            event.type,
        )
```

#### Fix 2: stop() — eliminate drain race
```python
async def stop(self) -> None:
    """Stop the bus. Drains remaining events before stopping."""
    self._running = False
    if self._task:
        # Let the loop exit cleanly first
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        # THEN drain remaining events
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch(event)
            except asyncio.QueueEmpty:
                break
    logger.info("Event bus stopped")
```

Everything else from spec Phase A (Event dataclass, EventBus class, on(), emit(), set_db_persister(), _process_loop(), _dispatch()) is correct — use as-is.

---

### Phase 2: Shared Utilities

**File:** `nous/handlers/__init__.py` (NEW)

```python
"""Event handlers for Nous.

Handlers listen to bus events and react asynchronously.
Each handler registers itself on specific event types during __init__.
"""

from nous.config import Settings


def build_anthropic_headers(settings: Settings) -> dict[str, str]:
    """Build auth headers for Anthropic API calls.

    Shared by all handlers that make LLM calls (episode_summarizer,
    fact_extractor, sleep_handler).
    """
    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    api_key = getattr(settings, "anthropic_auth_token", None) or getattr(
        settings, "anthropic_api_key", None
    )
    if api_key and "sk-ant-oat" in api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
        headers["anthropic-dangerous-direct-browser-access"] = "true"
    else:
        headers["x-api-key"] = api_key or ""
    return headers
```

---

### Phase 3: Wire Bus Into Existing Code

**Files:** `nous/cognitive/layer.py`, `nous/main.py`, `nous/runtime/runner.py`

#### 3A: CognitiveLayer — Add bus parameter + transcript capture

```python
# layer.py __init__ — CORRECT signature (preserves identity_prompt)
def __init__(
    self,
    brain: Brain,
    heart: Heart,
    settings: Settings,
    identity_prompt: str = "",
    *,
    bus: EventBus | None = None,
):
    ...existing init...
    self._bus = bus
```

**pre_turn** — add transcript capture + user_id passthrough:
```python
# Add user_id, user_display_name params
async def pre_turn(
    self,
    agent_id: str,
    session_id: str,
    user_input: str,
    session: AsyncSession | None = None,
    conversation_messages: list | None = None,
    user_id: str | None = None,           # NEW
    user_display_name: str | None = None,  # NEW
):
    ...
    # After meta tracking, add transcript
    meta.transcript.append(f"User: {user_input[:500]}")

    # When creating episode, pass user info
    # In the episode creation section, add to EpisodeInput:
    #   user_id=user_id, user_display_name=user_display_name
```

**post_turn** — add transcript capture + bus emit with backward compat:
```python
# After existing meta tracking
meta.transcript.append(f"Assistant: {turn_result.response_text[:500]}")

# Replace Brain.emit_event with bus.emit (BACKWARD COMPAT)
if self._bus:
    await self._bus.emit(Event(
        type="turn_completed",
        agent_id=agent_id,
        session_id=session_id,
        data={...existing data...},
    ))
else:
    await self._brain.emit_event("turn_completed", {...}, session=session)
```

**end_session** — pass transcript in event + bus emit with backward compat:
```python
# Build transcript from metadata
transcript_text = "\n\n".join(meta.transcript) if meta else ""

if self._bus:
    await self._bus.emit(Event(
        type="session_ended",
        agent_id=agent_id,
        session_id=session_id,
        data={
            "episode_id": str(episode_id) if episode_id else None,
            "transcript": transcript_text,
            "reflection": reflection[:200] if reflection else None,
        },
    ))
else:
    await self._brain.emit_event("session_ended", {...}, session=session)
```

#### 3B: Runner — Add user_id params

```python
# runner.py — run_turn signature
async def run_turn(
    self,
    session_id: str,
    user_message: str,
    agent_id: str | None = None,
    user_id: str | None = None,           # NEW
    user_display_name: str | None = None,  # NEW
) -> ...:
    # Pass to cognitive.pre_turn:
    turn_context = await self._cognitive.pre_turn(
        _agent_id, session_id, user_message,
        user_id=user_id, user_display_name=user_display_name,
        ...
    )

# runner.py — stream_chat signature (same additions)
async def stream_chat(
    self,
    session_id: str,
    user_message: str,
    agent_id: str | None = None,
    user_id: str | None = None,           # NEW
    user_display_name: str | None = None,  # NEW
) -> ...:
    # Same passthrough
```

#### 3C: Main.py — Create bus, register handlers, lifecycle

```python
# In create_components() — after Brain and Heart are created

from nous.events import EventBus, Event

# Create bus (only if enabled)
bus = None
if settings.event_bus_enabled:
    bus = EventBus()

    # DB persistence adapter (P0-1 fix: correct signature)
    async def persist_to_db(event: Event) -> None:
        data = {**event.data}
        if event.session_id:
            data["session_id"] = event.session_id
        await brain.emit_event(event.type, data)

    bus.set_db_persister(persist_to_db)

    # Register handlers (Phase 4 handlers)
    handler_http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10)
    )
    if settings.episode_summary_enabled:
        EpisodeSummarizer(heart, settings, bus, handler_http)
    if settings.fact_extraction_enabled:
        FactExtractor(heart, settings, bus, handler_http)

    # Session timeout monitor (P0-10 fix: inject cognitive reference)
    session_monitor = SessionTimeoutMonitor(bus, settings, cognitive=cognitive)

    if settings.sleep_enabled:
        SleepHandler(brain, heart, settings, bus, handler_http)

    # Start bus + monitor
    await bus.start()
    await session_monitor.start()

# Pass bus to Cognitive Layer (P0-2/P0-3 fix: preserve identity_prompt)
cognitive = CognitiveLayer(brain, heart, settings, settings.identity_prompt, bus=bus)

# Add to components dict
components["bus"] = bus
components["session_monitor"] = session_monitor  # if bus enabled
components["handler_http"] = handler_http  # if bus enabled
```

NOTE: `cognitive` must be created BEFORE `session_monitor` since monitor needs cognitive reference. Adjust creation order:
1. Brain, Heart
2. CognitiveLayer (with bus)
3. EventBus + handlers + monitor (with cognitive reference)

```python
# In shutdown_components()
if components.get("session_monitor"):
    await components["session_monitor"].stop()
if components.get("bus"):
    await components["bus"].stop()
if components.get("handler_http"):
    await components["handler_http"].aclose()
```

---

### Phase 4: Event Handlers (4 handlers)

**Parallel with Phase 5 (tests)**

#### 4A: Episode Summarizer (`nous/handlers/episode_summarizer.py`)

Use spec Phase C code with these fixes:
- Use `build_anthropic_headers(settings)` from `nous/handlers/__init__.py` instead of inline auth
- Fix: check `episode.structured_summary is not None` to skip re-summarization (not `episode.summary`)
- Truncation logic is correct as-is

#### 4B: Fact Extractor (`nous/handlers/fact_extractor.py`)

Use spec Phase D code with these fixes:
- **P0-7 fix:** Use `.score` not `.similarity`:
  ```python
  if existing and existing[0].score is not None and existing[0].score > 0.65:
  ```
- **P1-8 fix:** Pass category:
  ```python
  fact_input = FactInput(
      subject=fact.get("subject", "unknown"),
      content=content,
      source="fact_extractor",
      confidence=confidence,
      category=fact.get("category"),  # NEW
  )
  ```
- Use `build_anthropic_headers(settings)` from shared utility

#### 4C: Session Timeout Monitor (`nous/handlers/session_monitor.py`)

**MAJOR REWRITE from spec** (P0-10 fix):

The monitor must call `cognitive.end_session()` instead of emitting raw `session_ended`:

```python
class SessionTimeoutMonitor:
    def __init__(self, bus: EventBus, settings: Settings, *, cognitive=None):
        self._bus = bus
        self._settings = settings
        self._cognitive = cognitive  # NEW: reference to CognitiveLayer
        self._last_activity: dict[str, float] = {}
        self._last_agent: dict[str, str] = {}
        self._global_last_activity: float = time.monotonic()
        self._sleep_emitted: bool = False
        self._task: asyncio.Task | None = None

        bus.on("turn_completed", self.on_activity)
        bus.on("message_received", self.on_activity)
        bus.on("session_ended", self._on_session_ended)  # P1-4 fix: cleanup

    async def _on_session_ended(self, event: Event) -> None:
        """Clean up tracking when session ends (explicit or timeout)."""
        if event.session_id:
            self._last_activity.pop(event.session_id, None)
            self._last_agent.pop(event.session_id, None)

    async def _check_timeouts(self) -> None:
        now = time.monotonic()
        expired = []
        for session_id, last in list(self._last_activity.items()):
            idle_seconds = now - last
            if idle_seconds > self._settings.session_idle_timeout:
                agent_id = self._last_agent.get(session_id, "unknown")
                logger.info("Session %s idle for %ds, triggering end_session",
                           session_id, int(idle_seconds))

                # P0-10 fix: call cognitive.end_session() which has
                # episode_id, transcript, and does full cleanup
                if self._cognitive:
                    try:
                        await self._cognitive.end_session(
                            agent_id=agent_id,
                            session_id=session_id,
                            reflection=None,
                        )
                    except Exception:
                        logger.exception("Failed to end timed-out session %s", session_id)

                expired.append(session_id)

        for sid in expired:
            self._last_activity.pop(sid, None)
            self._last_agent.pop(sid, None)

        # Global sleep check — same as spec
        global_idle = now - self._global_last_activity
        if (global_idle > self._settings.sleep_timeout
                and not self._sleep_emitted
                and not self._last_activity):
            await self._bus.emit(Event(
                type="sleep_started",
                agent_id="system",
                data={"idle_seconds": int(global_idle)},
            ))
            self._sleep_emitted = True
```

#### 4D: Sleep Handler (`nous/handlers/sleep_handler.py`)

**ARCHITECTURAL FIX** (P0-11): Handler must spawn separate asyncio.Task:

```python
class SleepHandler:
    def __init__(self, brain, heart, settings, bus, http_client=None):
        ...same as spec...
        self._sleep_task: asyncio.Task | None = None
        bus.on("sleep_started", self.handle)
        bus.on("message_received", self._on_wake)

    async def handle(self, event: Event) -> None:
        """Spawn sleep work as background task — return immediately to unblock bus."""
        if self._sleeping:
            return  # Already sleeping
        self._sleep_task = asyncio.create_task(
            self._run_sleep(event), name="sleep-work"
        )

    async def _run_sleep(self, event: Event) -> None:
        """Actual sleep work — runs as independent task, NOT blocking bus."""
        self._sleeping = True
        self._interrupted = False
        phases_completed = []
        try:
            # Phase ordering: free first, LLM last (same as spec)
            if not self._interrupted:
                await self._phase_review_decisions()
                phases_completed.append("review")
            if not self._interrupted:
                await self._phase_prune()
                phases_completed.append("prune")
            if not self._interrupted:
                await self._phase_compress()
                phases_completed.append("compress")
            if not self._interrupted:
                await self._phase_reflect()
                phases_completed.append("reflect")
            if not self._interrupted:
                await self._phase_generalize()
                phases_completed.append("generalize")

            await self._bus.emit(Event(
                type="sleep_completed",
                agent_id=event.agent_id,
                data={"phases_completed": phases_completed, "interrupted": self._interrupted},
            ))
        except Exception:
            logger.exception("Sleep handler error")
        finally:
            self._sleeping = False
            self._sleep_task = None
```

Use `build_anthropic_headers(settings)` for LLM calls. Replace `search_episodes("")` with a proper recent-episodes query.

---

### Phase 5: Tests (~41 test cases)

**File:** `tests/test_event_bus.py` (NEW)

**Parallel with Phase 4.**

Test classes per spec Phase L, with additions:
- Test bus=None backward compat (events still persist via Brain.emit_event)
- Test sleep handler spawns task and returns immediately (bus not blocked)
- Test session monitor calls cognitive.end_session()
- Test FactExtractor dedup with .score attribute
- Test transcript accumulation in SessionMetadata

---

## File Change Summary

| File | Change | Phase |
|------|--------|-------|
| `nous/config.py` | Add 8 settings fields | 0A |
| `nous/cognitive/schemas.py` | Add transcript to SessionMetadata | 0B |
| `nous/heart/schemas.py` | Add fields to EpisodeInput/Detail/Summary | 0C |
| `nous/storage/models.py` | Add 3 columns to Episode ORM | 0D |
| `sql/init.sql` | Add columns + indexes to CREATE TABLE | 0E |
| `sql/migrations/006_event_bus.sql` | ALTER TABLE for existing installs | 0E |
| `nous/heart/episodes.py` | Add update_summary, map new columns | 0F |
| `nous/heart/heart.py` | Add update_episode_summary delegation | 0F |
| `nous/events.py` | **NEW** — Event, EventBus | 1 |
| `nous/handlers/__init__.py` | **NEW** — build_anthropic_headers | 2 |
| `nous/cognitive/layer.py` | Add bus, transcript, user_id, backward compat | 3A |
| `nous/runtime/runner.py` | Add user_id/user_display_name params | 3B |
| `nous/main.py` | Create bus, handlers, lifecycle | 3C |
| `nous/handlers/episode_summarizer.py` | **NEW** — F010.1 | 4A |
| `nous/handlers/fact_extractor.py` | **NEW** — F010.4 (dedup fixed) | 4B |
| `nous/handlers/session_monitor.py` | **NEW** — calls cognitive.end_session | 4C |
| `nous/handlers/sleep_handler.py` | **NEW** — spawns asyncio.Task | 4D |
| `tests/test_event_bus.py` | **NEW** — 41+ test cases | 5 |

## Implementation Team

| Agent | Role | Work | Depends On |
|-------|------|------|------------|
| core-eng-006 | Core Engineer | Phases 0, 1, 2, 3 (bus + config + schemas + wiring) | — |
| handler-eng-006 | Handler Engineer | Phase 4 (all 4 handlers) | Phase 0-2 complete |
| test-eng-006 | Test Engineer | Phase 5 (41+ tests) | Phase 0-1 complete |

**Parallelization:** After core-eng-006 completes Phases 0-2, handler-eng-006 and test-eng-006 start in parallel. Core-eng-006 continues with Phase 3 wiring concurrently.

## Execution Order

```
Phase 0 (pre-work)  ────────┐
Phase 1 (EventBus core) ────┤
Phase 2 (shared utils) ─────┤
                             ├──► Phase 4 (handlers) ──► Code Review
Phase 3 (wiring) ───────────┤       ↕ parallel
                             └──► Phase 5 (tests)    ──► Test Run
```
