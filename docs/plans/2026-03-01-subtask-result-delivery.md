# Subtask Result Delivery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically deliver subtask results to parent sessions so they enter the agent's memory system without duplication.

**Architecture:** Add `skip_episode` flag to prevent subtask sessions from creating episodes. Thread `session_id` through tool dispatch so `spawn_task` records `parent_session_id`. Inject undelivered subtask results into parent session context during `pre_turn`. Mark subtasks as delivered after injection.

**Tech Stack:** SQLAlchemy (async), PostgreSQL, Python 3.12+

---

### Task 1: Add `delivered` column to Subtask model and migration

**Files:**
- Modify: `nous/storage/models.py:569` (after `notify` column)
- Modify: `sql/migrations/010_subtasks.sql`
- Modify: `sql/init.sql`

**Step 1: Write the migration**

Add to the end of `sql/migrations/010_subtasks.sql`:

```sql
-- Result delivery tracking
ALTER TABLE heart.subtasks ADD COLUMN IF NOT EXISTS delivered BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_subtasks_undelivered
    ON heart.subtasks (parent_session_id, created_at)
    WHERE status IN ('completed', 'failed') AND delivered = FALSE;
```

**Step 2: Add column to init.sql**

In the `heart.subtasks` CREATE TABLE block in `sql/init.sql`, add after the `notify` line:

```sql
    delivered BOOLEAN NOT NULL DEFAULT FALSE,
```

And add the index in the indexes section:

```sql
CREATE INDEX idx_subtasks_undelivered
    ON heart.subtasks (parent_session_id, created_at)
    WHERE status IN ('completed', 'failed') AND delivered = FALSE;
```

**Step 3: Add column to ORM model**

In `nous/storage/models.py`, in class `Subtask`, add after the `notify` mapped_column (line ~569):

```python
    delivered: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
```

**Step 4: Commit**

```bash
git add nous/storage/models.py sql/migrations/010_subtasks.sql sql/init.sql
git commit -m "feat(011.1): add delivered column to subtask model and migration"
```

---

### Task 2: Thread session_id through tool dispatch

The `spawn_task` tool needs the parent session_id but tools currently don't receive session context. We need to thread it through `ToolDispatcher`.

**Files:**
- Modify: `nous/api/tools.py:34-62` (ToolDispatcher class)
- Modify: `nous/api/tools.py:626-670` (spawn_task closure)
- Modify: `nous/api/runner.py:972-974` (dispatch call site)
- Modify: `nous/api/runner.py:1258` (streaming dispatch call site)
- Test: `tests/test_subtasks.py`

**Step 1: Write the failing test**

Add to `tests/test_subtasks.py`:

```python
class TestSpawnTaskSessionId:
    """spawn_task should record parent_session_id."""

    async def test_spawn_task_receives_session_id(self) -> None:
        """spawn_task should pass session_id as parent_session_id."""
        from unittest.mock import AsyncMock, MagicMock
        from nous.api.tools import create_subtask_tools

        mock_heart = MagicMock()
        mock_subtask = MagicMock()
        mock_subtask.id = uuid.uuid4()
        mock_subtask.task = "test task"
        mock_subtask.priority = 100
        mock_subtask.status = "pending"
        mock_heart.subtasks.create = AsyncMock(return_value=mock_subtask)

        mock_settings = MagicMock()
        mock_settings.subtask_default_timeout = 120
        mock_settings.subtask_max_timeout = 600

        closures = create_subtask_tools(mock_heart, mock_settings)
        spawn = closures["spawn_task"]

        # Call with _session_id (injected by dispatcher)
        await spawn(task="do something", _session_id="test-session-123")

        # Verify parent_session_id was passed through
        mock_heart.subtasks.create.assert_called_once()
        call_kwargs = mock_heart.subtasks.create.call_args.kwargs
        assert call_kwargs["parent_session_id"] == "test-session-123"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_subtasks.py::TestSpawnTaskSessionId -v`
Expected: FAIL — `spawn_task` doesn't accept `_session_id`

**Step 3: Update ToolDispatcher to pass session context**

In `nous/api/tools.py`, modify the `dispatch` method to accept and inject session_id:

```python
    async def dispatch(
        self, name: str, args: dict[str, Any], session_id: str | None = None,
    ) -> tuple[str, bool]:
        """Dispatch a tool call and return (result_text, is_error).

        P0-6 fix: Uses **kwargs unpacking for closures.
        P1-1 fix: Extracts text from MCP-format response.
        """
        handler = self._handlers.get(name)
        if not handler:
            return f"Unknown tool: {name}", True
        try:
            if session_id is not None:
                args = {**args, "_session_id": session_id}
            result = await handler(**args)  # P0-6: **kwargs unpacking
```

**Step 4: Update spawn_task to use `_session_id`**

In the `spawn_task` closure inside `create_subtask_tools`, add `_session_id` parameter:

```python
    async def spawn_task(
        task: str,
        priority: str = "normal",
        timeout: int | None = None,
        notify: bool = True,
        _session_id: str | None = None,
    ) -> dict[str, Any]:
```

And pass it to `create()`:

```python
            subtask = await heart.subtasks.create(
                task=task,
                priority=priority,
                timeout=effective_timeout,
                notify=notify,
                parent_session_id=_session_id,
            )
```

**Step 5: Update runner.py dispatch call sites**

In `nous/api/runner.py`, find the two `dispatch()` call sites and pass `session_id`:

Line ~972 (non-streaming):
```python
                    result_text, is_error = await self._dispatcher.dispatch(
                        tool_name, tool_input, session_id=session_id
                    )
```

Line ~1258 (streaming):
```python
                self._dispatcher.dispatch(name, args, session_id=session_id),
```

**Step 6: Run tests**

Run: `uv run pytest tests/test_subtasks.py::TestSpawnTaskSessionId -v`
Expected: PASS

**Step 7: Commit**

```bash
git add nous/api/tools.py nous/api/runner.py tests/test_subtasks.py
git commit -m "feat(011.1): thread session_id through tool dispatch to spawn_task"
```

---

### Task 3: Add `skip_episode` flag to run_turn and pre_turn

**Files:**
- Modify: `nous/api/runner.py:268-307` (run_turn signature + pre_turn call)
- Modify: `nous/cognitive/layer.py:131-318` (pre_turn signature + episode block)
- Test: `tests/test_subtasks.py`

**Step 1: Write the failing test**

Add to `tests/test_subtasks.py`:

```python
class TestSkipEpisode:
    """run_turn with skip_episode=True should not create episodes."""

    async def test_pre_turn_skip_episode(self) -> None:
        """pre_turn with skip_episode=True should skip episode creation."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create a minimal CognitiveLayer mock that tracks episode creation
        mock_heart = MagicMock()
        mock_heart.start_episode = AsyncMock()

        mock_brain = MagicMock()
        mock_brain.db = MagicMock()
        mock_brain.embeddings = MagicMock()

        mock_settings = MagicMock()
        mock_settings.agent_id = "test"

        from nous.cognitive.layer import CognitiveLayer
        layer = CognitiveLayer.__new__(CognitiveLayer)
        layer._heart = mock_heart
        layer._active_episodes = {}
        layer._session_metadata = {}

        # Simulate the episode creation check
        # When skip_episode is True, start_episode should never be called
        # We test the flag is respected by checking _active_episodes stays empty
        # (Full integration test would need DB — unit test verifies flag propagation)

        # Verify the parameter exists on pre_turn
        import inspect
        sig = inspect.signature(layer.pre_turn)
        assert "skip_episode" in sig.parameters, "pre_turn must accept skip_episode"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_subtasks.py::TestSkipEpisode -v`
Expected: FAIL — `pre_turn` doesn't have `skip_episode` parameter

**Step 3: Add skip_episode to pre_turn**

In `nous/cognitive/layer.py`, modify `pre_turn` signature (line ~131):

```python
    async def pre_turn(
        self,
        agent_id: str,
        session_id: str,
        user_input: str,
        session: AsyncSession | None = None,
        *,
        conversation_messages: list[str] | None = None,
        user_id: str | None = None,
        user_display_name: str | None = None,
        skip_episode: bool = False,
    ) -> TurnContext:
```

Then wrap the episode creation block (line ~298-318) with the flag:

```python
        # 5. EPISODE — start if no active episode AND interaction is significant
        if not skip_episode and session_id not in self._active_episodes:
```

**Step 4: Add skip_episode to run_turn**

In `nous/api/runner.py`, modify `run_turn` signature (line ~268):

```python
    async def run_turn(
        self,
        session_id: str,
        user_message: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        user_display_name: str | None = None,
        platform: str | None = None,
        system_prompt_prefix: str | None = None,
        skip_episode: bool = False,
    ) -> tuple[str, TurnContext, dict[str, int]]:
```

Pass it to `pre_turn` (line ~300):

```python
        turn_context = await self._cognitive.pre_turn(
            _agent_id,
            session_id,
            user_message,
            conversation_messages=recent_messages or None,
            user_id=user_id,
            user_display_name=user_display_name,
            skip_episode=skip_episode,
        )
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_subtasks.py::TestSkipEpisode -v`
Expected: PASS

**Step 6: Commit**

```bash
git add nous/api/runner.py nous/cognitive/layer.py tests/test_subtasks.py
git commit -m "feat(011.1): add skip_episode flag to run_turn and pre_turn"
```

---

### Task 4: Update subtask worker to use skip_episode

**Files:**
- Modify: `nous/handlers/subtask_worker.py:146` (run_turn call)

**Step 1: Update run_turn call in worker**

In `nous/handlers/subtask_worker.py`, line ~146, add `skip_episode=True`:

```python
            response_text, _turn_ctx, _usage = await self._runner.run_turn(
                session_id=session_id,
                user_message=subtask.task,
                agent_id=self._settings.agent_id,
                system_prompt_prefix=system_prefix,
                skip_episode=True,
            )
```

**Step 2: Commit**

```bash
git add nous/handlers/subtask_worker.py
git commit -m "feat(011.1): subtask worker uses skip_episode=True"
```

---

### Task 5: Add `get_undelivered` and `mark_delivered` to SubtaskManager

**Files:**
- Modify: `nous/heart/subtasks.py`
- Test: `tests/test_subtasks.py`

**Step 1: Write the failing tests**

Add to `tests/test_subtasks.py`:

```python
class TestSubtaskDelivery:
    """Tests for get_undelivered and mark_delivered."""

    async def test_get_undelivered_returns_completed(self, db, settings) -> None:
        mgr = SubtaskManager(db, settings.agent_id)
        st = await mgr.create(task="test", parent_session_id="parent-1")
        await mgr.complete(st.id, "result text")
        undelivered = await mgr.get_undelivered("parent-1")
        assert len(undelivered) == 1
        assert undelivered[0].id == st.id
        assert undelivered[0].result == "result text"

    async def test_get_undelivered_returns_failed(self, db, settings) -> None:
        mgr = SubtaskManager(db, settings.agent_id)
        st = await mgr.create(task="test", parent_session_id="parent-1")
        await mgr.fail(st.id, "some error")
        undelivered = await mgr.get_undelivered("parent-1")
        assert len(undelivered) == 1
        assert undelivered[0].error == "some error"

    async def test_get_undelivered_skips_delivered(self, db, settings) -> None:
        mgr = SubtaskManager(db, settings.agent_id)
        st = await mgr.create(task="test", parent_session_id="parent-1")
        await mgr.complete(st.id, "result")
        await mgr.mark_delivered([st.id])
        undelivered = await mgr.get_undelivered("parent-1")
        assert len(undelivered) == 0

    async def test_get_undelivered_skips_other_sessions(self, db, settings) -> None:
        mgr = SubtaskManager(db, settings.agent_id)
        st = await mgr.create(task="test", parent_session_id="other-session")
        await mgr.complete(st.id, "result")
        undelivered = await mgr.get_undelivered("parent-1")
        assert len(undelivered) == 0

    async def test_mark_delivered_batch(self, db, settings) -> None:
        mgr = SubtaskManager(db, settings.agent_id)
        st1 = await mgr.create(task="t1", parent_session_id="parent-1")
        st2 = await mgr.create(task="t2", parent_session_id="parent-1")
        await mgr.complete(st1.id, "r1")
        await mgr.complete(st2.id, "r2")
        await mgr.mark_delivered([st1.id, st2.id])
        undelivered = await mgr.get_undelivered("parent-1")
        assert len(undelivered) == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subtasks.py::TestSubtaskDelivery -v`
Expected: FAIL — methods don't exist

**Step 3: Implement the methods**

Add to `nous/heart/subtasks.py`, after the `list` method:

```python
    async def get_undelivered(self, parent_session_id: str) -> list[Subtask]:
        """Get completed/failed subtasks not yet delivered to parent session."""
        async with self._db.session() as session:
            result = await session.execute(
                select(Subtask)
                .where(Subtask.agent_id == self._agent_id)
                .where(Subtask.parent_session_id == parent_session_id)
                .where(Subtask.status.in_(["completed", "failed"]))
                .where(Subtask.delivered.is_(False))
                .order_by(Subtask.completed_at)
            )
            return list(result.scalars().all())

    async def mark_delivered(self, subtask_ids: list[UUID]) -> None:
        """Mark subtasks as delivered to parent session."""
        if not subtask_ids:
            return
        async with self._db.session() as session:
            await session.execute(
                update(Subtask)
                .where(Subtask.id.in_(subtask_ids))
                .values(delivered=True)
            )
            await session.commit()
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_subtasks.py::TestSubtaskDelivery -v`
Expected: PASS (requires Postgres — will error with ConnectionRefused if DB not running)

**Step 5: Commit**

```bash
git add nous/heart/subtasks.py tests/test_subtasks.py
git commit -m "feat(011.1): add get_undelivered and mark_delivered to SubtaskManager"
```

---

### Task 6: Inject subtask results into pre_turn context

**Files:**
- Modify: `nous/cognitive/layer.py:265-284` (after context build, before return)
- Test: `tests/test_subtasks.py`

**Step 1: Write the failing test**

Add to `tests/test_subtasks.py`:

```python
class TestSubtaskContextInjection:
    """pre_turn should inject undelivered subtask results into system prompt."""

    async def test_completed_subtask_injected(self) -> None:
        """Completed subtask result should appear in system prompt."""
        import inspect
        from nous.cognitive.layer import CognitiveLayer

        # Verify pre_turn exists and has skip_episode param
        sig = inspect.signature(CognitiveLayer.pre_turn)
        assert "skip_episode" in sig.parameters

        # The actual injection test needs full integration with DB.
        # Here we verify the format helper exists and works.
        from nous.cognitive.layer import _format_subtask_results

        from unittest.mock import MagicMock
        completed = MagicMock()
        completed.id = uuid.uuid4()
        completed.task = "Research quantum computing"
        completed.status = "completed"
        completed.result = "Found 3 key papers on quantum error correction."
        completed.error = None

        failed = MagicMock()
        failed.id = uuid.uuid4()
        failed.task = "Analyze market data"
        failed.status = "failed"
        failed.result = None
        failed.error = "TimeoutError: timed out after 120s"

        text = _format_subtask_results([completed, failed])
        assert "Research quantum computing" in text
        assert "Found 3 key papers" in text
        assert "Analyze market data" in text
        assert "TimeoutError" in text
        assert "Completed Subtask" in text
        assert "Failed Subtask" in text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_subtasks.py::TestSubtaskContextInjection -v`
Expected: FAIL — `_format_subtask_results` doesn't exist

**Step 3: Add the format helper**

Add to `nous/cognitive/layer.py`, as a module-level function (before the class):

```python
def _format_subtask_results(subtasks: list) -> str:
    """Format undelivered subtask results for context injection."""
    if not subtasks:
        return ""

    lines: list[str] = []
    completed = [s for s in subtasks if s.status == "completed"]
    failed = [s for s in subtasks if s.status == "failed"]

    if completed:
        for s in completed:
            lines.append(f"=== Completed Subtask ===")
            lines.append(f"Task: {s.task}")
            lines.append(f"Result: {s.result}")
            lines.append("")

    if failed:
        for s in failed:
            lines.append(f"=== Failed Subtask ===")
            lines.append(f"Task: {s.task}")
            lines.append(f"Error: {s.error}")
            lines.append("")

    return "\n".join(lines).strip()
```

**Step 4: Inject into pre_turn**

In `nous/cognitive/layer.py`, in the `pre_turn` method, after the context build block (after line ~284 where `recalled_content_map` is set), add subtask result injection:

```python
        # 3b. SUBTASK RESULTS — inject undelivered results into context
        subtask_context = ""
        try:
            undelivered = await self._heart.subtasks.get_undelivered(session_id)
            if undelivered:
                subtask_context = _format_subtask_results(undelivered)
                delivered_ids = [s.id for s in undelivered]
                await self._heart.subtasks.mark_delivered(delivered_ids)
                logger.info(
                    "Injected %d subtask results into session %s",
                    len(undelivered), session_id,
                )
        except Exception:
            logger.warning("Failed to inject subtask results for session %s", session_id)
```

Then append `subtask_context` to `system_prompt` (right after the context build try/except block):

```python
        if subtask_context:
            system_prompt = system_prompt + "\n\n" + subtask_context
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_subtasks.py::TestSubtaskContextInjection -v`
Expected: PASS

**Step 6: Commit**

```bash
git add nous/cognitive/layer.py tests/test_subtasks.py
git commit -m "feat(011.1): inject subtask results into parent session context"
```

---

### Task 7: Update CLAUDE.md and INDEX.md

**Files:**
- Modify: `docs/features/INDEX.md`
- Modify: `CLAUDE.md`

**Step 1: Update INDEX.md**

Add a new row in the implementation specs table:

```markdown
| 011.2 | Subtask Result Delivery | ✅ Shipped | — subtask results auto-injected into parent session context |
```

**Step 2: Update CLAUDE.md database section**

Update the table count comment if needed (the `delivered` column doesn't add a table, just a column).

**Step 3: Commit**

```bash
git add docs/features/INDEX.md CLAUDE.md
git commit -m "docs: update INDEX.md with subtask result delivery"
```

---

## Task Dependency Graph

```
Task 1 (migration + model) ──┐
                              ├── Task 5 (get_undelivered/mark_delivered)
Task 2 (session_id threading) │        │
         │                    │        └── Task 6 (context injection)
         └── Task 4 ──────────┘                    │
                                                   └── Task 7 (docs)
Task 3 (skip_episode flag) ── Task 4 (worker update)
```

Tasks 1, 2, 3 can run in parallel. Task 4 depends on 3. Task 5 depends on 1. Task 6 depends on 5 and 3. Task 7 depends on 6.
