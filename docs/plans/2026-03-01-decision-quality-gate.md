# Decision Quality Gate (009.5) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce decision noise from 43% to <5% via a three-layer quality gate.

**Architecture:** Expand `_is_informational()` patterns + action report detection (Layer 1), add embedding-based dedup in `deliberation.start()` (Layer 2), add content quality rules at `finalize()` with hard delete (Layer 3). Remove `task` from auto-deliberation frames.

**Tech Stack:** Python 3.12+, SQLAlchemy 2.0 async, pgvector embeddings, pytest-asyncio

**Design deviation:** The design doc says soft-delete (`active=false`) but `brain.decisions` has no `active` column. Adding one would require schema migration + updating every decision query. Use existing `Brain.delete()` (hard delete) instead — `logger.info` provides the audit trail. The existing `_delete()` method already handles cascading cleanup.

---

### Task 1: Remove `task` from Deliberation Frames

**Files:**
- Modify: `nous/cognitive/deliberation.py:21`
- Modify: `tests/test_deliberation.py:152-157`

**Step 1: Update the existing test to expect `task` frame NOT to trigger deliberation**

In `tests/test_deliberation.py`, change `test_should_deliberate_task`:

```python
async def test_should_deliberate_task(delib):
    """Task frame does NOT trigger deliberation (009.5)."""
    frame = _frame("task", frame_name="Task Execution")
    result = await delib.should_deliberate(frame)
    assert result is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deliberation.py::test_should_deliberate_task -v`
Expected: FAIL — `assert True is False`

**Step 3: Update `_DELIBERATION_FRAMES`**

In `nous/cognitive/deliberation.py` line 21:

```python
# Frames that trigger deliberation (D8, 009.5: removed task)
_DELIBERATION_FRAMES = {"decision", "debug"}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deliberation.py::test_should_deliberate_task -v`
Expected: PASS

**Step 5: Run full deliberation test suite**

Run: `uv run pytest tests/test_deliberation.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add nous/cognitive/deliberation.py tests/test_deliberation.py
git commit -m "feat(009.5): remove task from auto-deliberation frames"
```

---

### Task 2: Expand `_INFO_PATTERNS` with New Chat Fragment Patterns

**Files:**
- Modify: `nous/cognitive/layer.py:557-576`
- Modify: `tests/test_cognitive_layer.py`

**Step 1: Write failing tests for new patterns**

Add to `tests/test_cognitive_layer.py`:

```python
# ---------------------------------------------------------------------------
# 009.5: Expanded informational patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response_text", [
    "Done! ✅ PR #90 created successfully",
    "done. Moving on to the next task.",
    "Completed! All tests passing.",
    "Finished! The implementation is ready.",
    "On it! 🚀 Subtask is running...",
    "Created! The new file is at src/main.py",
    "Pushed to main branch.",
    "Review complete — spec scores 8/10.",
    "Task is running in the background.",
    "Now let me check the database schema.",
    "Next I'll implement the API endpoint.",
    "Moving on to the test suite.",
    "Let me check if the tests pass.",
    "Let me look at the error logs.",
    "I'll start with the database migration.",
    "Starting with the schema changes.",
    "Here's the result of the analysis.",
    "Here are the results from the query.",
    "PR #42 has been merged.",
    "PR created and ready for review.",
    "Spec scores look good across the board.",
])
async def test_is_informational_009_5_patterns(cognitive_layer, response_text):
    """009.5: New chat fragment patterns are detected as informational."""
    turn_result = _turn_result(response_text=response_text)
    assert cognitive_layer._is_informational(turn_result) is True
```

Note: This test uses the existing `_turn_result` helper and `cognitive_layer` fixture from the test file. Check the existing test file for these helpers and adapt if needed.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cognitive_layer.py::test_is_informational_009_5_patterns -v`
Expected: FAIL — most patterns not matched

**Step 3: Add new patterns to `_INFO_PATTERNS`**

In `nous/cognitive/layer.py`, extend `_INFO_PATTERNS` after the existing entries (after line 576):

```python
_INFO_PATTERNS = [
    # Status & inventory
    "current status", "available tools", "here's what",
    "here is what", "here are the", "summary of",
    # Memory recall
    "i remember", "my memory", "what i know",
    "i recall", "from memory", "i found",
    # Git / repo status
    "repo pulled", "repo is at", "git pull",
    "latest commit", "new branch", "new pr",
    "commits since", "merged to main",
    # Acknowledgment / confirmation
    "got it", "understood", "noted", "will do",
    "sure thing", "okay,", "alright,",
    # Simple answers
    "the answer is", "it means", "this is because",
    "that's correct", "you're right",
    # Lists / enumerations
    "here's a list", "the following",
    # 009.5: Completion / status updates
    "done!", "done.", "completed!", "finished!",
    "on it!", "created!", "pushed to",
    "review complete", "spec scores", "task is running",
    # 009.5: Transition phrases
    "now let me", "next i'll", "moving on to",
    "let me check", "let me look",
    "i'll start", "starting with",
    # 009.5: Report phrases
    "here's the result", "here are the results",
    "pr #", "pr created",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cognitive_layer.py::test_is_informational_009_5_patterns -v`
Expected: PASS

**Step 5: Run full cognitive layer test suite**

Run: `uv run pytest tests/test_cognitive_layer.py -v`
Expected: All pass (no regressions)

**Step 6: Commit**

```bash
git add nous/cognitive/layer.py tests/test_cognitive_layer.py
git commit -m "feat(009.5): expand informational patterns for chat fragments"
```

---

### Task 3: Add `_is_action_report()` Method

**Files:**
- Modify: `nous/cognitive/layer.py` (add method near `_is_informational`)
- Modify: `tests/test_cognitive_layer.py`

**Step 1: Write failing tests**

Add to `tests/test_cognitive_layer.py`:

```python
# ---------------------------------------------------------------------------
# 009.5: Action report detection
# ---------------------------------------------------------------------------

async def test_is_action_report_with_tools_and_markers(cognitive_layer):
    """009.5: Response with tools + 2+ report markers is action report."""
    turn_result = _turn_result(
        response_text="Done! I've created the file and updated the config.",
        tool_results=[_tool_result("write_file")],
    )
    assert cognitive_layer._is_action_report(turn_result) is True


async def test_is_action_report_no_tools(cognitive_layer):
    """009.5: No tool results = not an action report."""
    turn_result = _turn_result(
        response_text="Done! I've created the file and updated the config.",
    )
    assert cognitive_layer._is_action_report(turn_result) is False


async def test_is_action_report_one_marker(cognitive_layer):
    """009.5: Single report marker is not enough."""
    turn_result = _turn_result(
        response_text="I've created the file. Now let me think about the architecture.",
        tool_results=[_tool_result("write_file")],
    )
    assert cognitive_layer._is_action_report(turn_result) is False


async def test_is_informational_delegates_to_action_report(cognitive_layer):
    """009.5: _is_informational catches action reports via _is_action_report."""
    turn_result = _turn_result(
        response_text="Fixed the bug and committed the changes. Deployed to staging.",
        tool_results=[_tool_result("bash")],
    )
    assert cognitive_layer._is_informational(turn_result) is True
```

Note: `_tool_result` helper needs to create a mock/simple object with `tool_name` and `error` attributes. Check existing test helpers — if none exists, create a minimal one:

```python
@dataclass
class _MockToolResult:
    tool_name: str
    error: str | None = None

def _tool_result(name: str, error: str | None = None):
    return _MockToolResult(tool_name=name, error=error)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cognitive_layer.py::test_is_action_report_with_tools_and_markers -v`
Expected: FAIL — `_is_action_report` not defined

**Step 3: Implement `_is_action_report()`**

Add to `CognitiveLayer` class in `nous/cognitive/layer.py`, after `_is_informational()`:

```python
# 009.5: Report markers for action report detection
_ACTION_REPORT_MARKERS = [
    "done", "created", "updated", "fixed", "merged",
    "pushed", "committed", "deployed", "sent", "saved",
    "completed", "finished", "resolved", "applied",
]

def _is_action_report(self, turn_result: TurnResult) -> bool:
    """Detect responses that report completed actions, not decisions (009.5).

    Pattern: tool calls happened + response summarizes what was done.
    If 2+ report markers in first 300 chars after tool use -> action report.
    """
    if not turn_result.tool_results:
        return False

    response_lower = turn_result.response_text[:300].lower()
    matches = sum(1 for m in self._ACTION_REPORT_MARKERS if m in response_lower)
    return matches >= 2
```

Wire into `_is_informational()` as check #5, before `return False`:

```python
    # 5. Action report: tools used + response summarizes what was done (009.5)
    if self._is_action_report(turn_result):
        return True

    return False
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cognitive_layer.py -k "action_report or informational_delegates" -v`
Expected: All pass

**Step 5: Run full test suite**

Run: `uv run pytest tests/test_cognitive_layer.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add nous/cognitive/layer.py tests/test_cognitive_layer.py
git commit -m "feat(009.5): add action report detection to informational filter"
```

---

### Task 4: Add `Brain.get_recent_decisions()` Method

**Files:**
- Modify: `nous/brain/brain.py` (add method after `_list_decisions`)
- Modify: `tests/test_brain.py` or create section in `tests/test_deliberation.py`

**Step 1: Write failing test**

Add to `tests/test_deliberation.py` (it already has a `brain` fixture):

```python
# ---------------------------------------------------------------------------
# 009.5: get_recent_decisions
# ---------------------------------------------------------------------------

async def test_get_recent_decisions_returns_recent(brain, session):
    """get_recent_decisions returns decisions created after cutoff."""
    from datetime import datetime, timedelta, UTC
    from nous.brain.schemas import RecordInput, ReasonInput

    # Create a decision
    inp = RecordInput(
        description="Use Redis for caching",
        confidence=0.8,
        category="architecture",
        stakes="medium",
        tags=["test"],
        reasons=[ReasonInput(type="analysis", text="Fast in-memory store")],
        session_id="test-session-1",
    )
    await brain.record(inp, session=session)

    cutoff = datetime.now(UTC) - timedelta(minutes=5)
    results = await brain.get_recent_decisions(
        brain.agent_id, since=cutoff, session_id="test-session-1", session=session,
    )
    assert len(results) == 1
    assert "Redis" in results[0].description


async def test_get_recent_decisions_filters_by_session(brain, session):
    """get_recent_decisions only returns decisions from the specified session."""
    from datetime import datetime, timedelta, UTC
    from nous.brain.schemas import RecordInput, ReasonInput

    for sid in ["session-a", "session-b"]:
        inp = RecordInput(
            description=f"Decision in {sid}",
            confidence=0.8,
            category="architecture",
            stakes="medium",
            tags=["test"],
            reasons=[ReasonInput(type="analysis", text="test reason")],
            session_id=sid,
        )
        await brain.record(inp, session=session)

    cutoff = datetime.now(UTC) - timedelta(minutes=5)
    results = await brain.get_recent_decisions(
        brain.agent_id, since=cutoff, session_id="session-a", session=session,
    )
    assert len(results) == 1
    assert "session-a" in results[0].description
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_deliberation.py::test_get_recent_decisions_returns_recent -v`
Expected: FAIL — `get_recent_decisions` not defined

**Step 3: Implement `get_recent_decisions()`**

Add to `Brain` class in `nous/brain/brain.py`, after `_list_decisions` (around line 158):

```python
async def get_recent_decisions(
    self,
    agent_id: str,
    since: datetime,
    limit: int = 5,
    session_id: str | None = None,
    session: AsyncSession | None = None,
) -> list[DecisionSummary]:
    """Fetch recent decisions since a cutoff time, optionally scoped to session.

    Returns decisions with descriptions and embeddings for dedup comparison (009.5).
    """
    if session is None:
        async with self.db.session() as session:
            return await self._get_recent_decisions(agent_id, since, limit, session_id, session)
    return await self._get_recent_decisions(agent_id, since, limit, session_id, session)

async def _get_recent_decisions(
    self,
    agent_id: str,
    since: datetime,
    limit: int,
    session_id: str | None,
    session: AsyncSession,
) -> list[DecisionSummary]:
    stmt = (
        select(Decision)
        .where(
            Decision.agent_id == agent_id,
            Decision.created_at >= since,
        )
        .order_by(Decision.created_at.desc())
        .limit(limit)
    )
    if session_id is not None:
        stmt = stmt.where(Decision.session_id == session_id)

    result = await session.execute(stmt)
    decisions = list(result.scalars().all())

    return [
        DecisionSummary(
            id=d.id,
            description=d.description,
            confidence=d.confidence,
            category=d.category,
            stakes=d.stakes,
            outcome=d.outcome or "pending",
            pattern=d.pattern,
            tags=[],  # Not needed for dedup
            created_at=d.created_at,
        )
        for d in decisions
    ]
```

Add `datetime` to imports at top of `brain.py` if not already present.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_deliberation.py -k "get_recent" -v`
Expected: All pass

**Step 5: Commit**

```bash
git add nous/brain/brain.py tests/test_deliberation.py
git commit -m "feat(009.5): add Brain.get_recent_decisions() for dedup"
```

---

### Task 5: Add Embedding-Based Dedup to `DeliberationEngine.start()`

**Files:**
- Modify: `nous/cognitive/deliberation.py`
- Modify: `tests/test_deliberation.py`

**Step 1: Write failing tests**

Add to `tests/test_deliberation.py`:

```python
# ---------------------------------------------------------------------------
# 009.5: Deduplication
# ---------------------------------------------------------------------------

async def test_start_dedup_blocks_duplicate(delib, brain, session):
    """009.5: start() returns None when a similar decision was recently recorded."""
    frame = _frame()

    # First call — should succeed
    id1 = await delib.start("nous-default", "evaluate database options", frame, session=session)
    assert id1 is not None

    # Second call with same description — should be deduped
    id2 = await delib.start(
        "nous-default", "evaluate database options", frame,
        session_id="test-session", session=session,
    )
    # Without embeddings, falls back to keyword overlap
    # Identical text -> containment = 1.0 > 0.5 threshold -> duplicate
    assert id2 is None


async def test_start_dedup_allows_different(delib, brain, session):
    """009.5: start() allows decisions with different descriptions."""
    frame = _frame()

    await delib.start("nous-default", "evaluate database options", frame, session=session)

    # Different topic — should NOT be deduped
    id2 = await delib.start(
        "nous-default", "choose frontend framework for dashboard", frame,
        session_id="test-session", session=session,
    )
    assert id2 is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_deliberation.py -k "dedup" -v`
Expected: FAIL — `start()` signature doesn't accept `session_id`

**Step 3: Implement dedup in `DeliberationEngine`**

Update `nous/cognitive/deliberation.py`:

```python
"""Deliberation engine — manages the deliberation lifecycle for a turn.

Thin wrapper around Brain.record(), Brain.think(), Brain.update()
providing a clean start -> think -> finalize flow.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from nous.brain.brain import Brain
from nous.brain.embeddings import EmbeddingProvider
from nous.brain.schemas import ReasonInput, RecordInput
from nous.cognitive.schemas import FrameSelection

logger = logging.getLogger(__name__)

# Frames that trigger deliberation (D8, 009.5: removed task)
_DELIBERATION_FRAMES = {"decision", "debug"}

# 009.5: Dedup thresholds
_DEDUP_WINDOW_MINUTES = 5
_DEDUP_EMBEDDING_THRESHOLD = 0.85
_DEDUP_KEYWORD_THRESHOLD = 0.5


class DeliberationEngine:
    """Manages the deliberation lifecycle for a single turn."""

    def __init__(self, brain: Brain) -> None:
        self._brain = brain

    async def start(
        self,
        agent_id: str,
        description: str,
        frame: FrameSelection,
        session_id: str | None = None,
        session: AsyncSession | None = None,
    ) -> str | None:
        """Begin deliberation — with dedup check (009.5).

        Returns decision_id as string, or None if duplicate detected.
        """
        # 009.5: Dedup check before creating decision
        if session_id and await self._is_duplicate(
            agent_id, description, session_id, session=session,
        ):
            logger.debug("Skipping duplicate deliberation: %s", description[:80])
            return None

        record_input = RecordInput(
            description=f"Plan: {description}",
            confidence=0.5,
            category=frame.default_category or "process",
            stakes=frame.default_stakes or "low",
            tags=[frame.frame_id],
            reasons=[
                ReasonInput(
                    type="analysis",
                    text=f"Frame '{frame.frame_name}' triggered deliberation for: {description[:100]}",
                )
            ],
            session_id=session_id,
        )

        detail = await self._brain.record(record_input, session=session)
        return str(detail.id)

    async def _is_duplicate(
        self,
        agent_id: str,
        description: str,
        session_id: str,
        session: AsyncSession | None = None,
    ) -> bool:
        """Check if a similar decision was recorded recently (009.5).

        Uses embedding similarity (0.85 threshold) with keyword fallback (0.5).
        Scoped to session_id to prevent cross-session suppression.
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=_DEDUP_WINDOW_MINUTES)
        recent = await self._brain.get_recent_decisions(
            agent_id, since=cutoff, session_id=session_id, session=session,
        )
        if not recent:
            return False

        # Try embedding-based comparison
        if self._brain.embeddings:
            try:
                desc_embedding = await self._brain.embeddings.embed(description)
                for decision in recent:
                    # Get stored embedding from DB
                    stored = await self._get_decision_embedding(decision.id, session=session)
                    if stored is not None:
                        sim = self._cosine_similarity(desc_embedding, stored)
                        if sim > _DEDUP_EMBEDDING_THRESHOLD:
                            return True
                return False
            except Exception:
                logger.debug("Embedding dedup failed, falling back to keyword overlap")

        # Fallback: keyword containment coefficient
        desc_words = set(re.findall(r"\b\w{3,}\b", description.lower()))
        for decision in recent:
            existing_words = set(re.findall(r"\b\w{3,}\b", decision.description.lower()))
            if not desc_words or not existing_words:
                continue
            intersection = desc_words & existing_words
            overlap = len(intersection) / min(len(desc_words), len(existing_words))
            if overlap > _DEDUP_KEYWORD_THRESHOLD:
                return True
        return False

    async def _get_decision_embedding(
        self, decision_id: UUID, session: AsyncSession | None = None,
    ) -> list[float] | None:
        """Fetch stored embedding for a decision."""
        from sqlalchemy import select
        from nous.storage.models import Decision
        if session is None:
            async with self._brain.db.session() as session:
                result = await session.execute(
                    select(Decision.embedding).where(Decision.id == decision_id)
                )
                return result.scalar_one_or_none()
        result = await session.execute(
            select(Decision.embedding).where(Decision.id == decision_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ... (think, finalize, delete, should_deliberate unchanged for this task)
```

**Step 4: Update `layer.py` to pass `session_id` to `start()`**

In `nous/cognitive/layer.py` around line 336, update the `start()` call:

```python
if await self._deliberation.should_deliberate(frame):
    decision_id = await self._deliberation.start(
        agent_id, user_input[:200], frame,
        session_id=session_id, session=session,
    )
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_deliberation.py -k "dedup" -v`
Expected: All pass

**Step 6: Run full test suite**

Run: `uv run pytest tests/test_deliberation.py tests/test_cognitive_layer.py -v`
Expected: All pass

**Step 7: Commit**

```bash
git add nous/cognitive/deliberation.py nous/cognitive/layer.py tests/test_deliberation.py
git commit -m "feat(009.5): add embedding-based dedup to deliberation start"
```

---

### Task 6: Add Quality Gate to `finalize()`

**Files:**
- Modify: `nous/cognitive/deliberation.py` (update `finalize()`)
- Modify: `tests/test_deliberation.py`

**Step 1: Write failing tests**

Add to `tests/test_deliberation.py`:

```python
# ---------------------------------------------------------------------------
# 009.5: Quality gate at finalize
# ---------------------------------------------------------------------------

async def test_finalize_rejects_short_description(delib, brain, session):
    """009.5: Description < 20 chars is rejected."""
    frame = _frame()
    decision_id = await delib.start("nous-default", "test short desc", frame, session=session)

    result = await delib.finalize(
        decision_id,
        description="Done!",  # 5 chars
        confidence=0.8,
        session=session,
    )
    assert result is None

    # Decision should be deleted
    detail = await brain.get(uuid.UUID(decision_id), session=session)
    assert detail is None


async def test_finalize_rejects_default_confidence_high_stakes(delib, brain, session):
    """009.5: confidence=0.5 + high stakes = never deliberated, reject."""
    frame = _frame(default_stakes="high")
    decision_id = await delib.start("nous-default", "some high stakes thing", frame, session=session)

    result = await delib.finalize(
        decision_id,
        description="Decided to restructure the entire database schema",
        confidence=0.5,  # Default — never updated from start()
        session=session,
    )
    assert result is None


async def test_finalize_rejects_chat_prefix(delib, brain, session):
    """009.5: Description starting with chat pattern is rejected."""
    frame = _frame()
    decision_id = await delib.start("nous-default", "test chat prefix", frame, session=session)

    result = await delib.finalize(
        decision_id,
        description="Got it! I'll implement the feature now.",
        confidence=0.8,
        session=session,
    )
    assert result is None


async def test_finalize_rejects_error_template(delib, brain, session):
    """009.5: Error template description is rejected."""
    frame = _frame()
    decision_id = await delib.start("nous-default", "test error template", frame, session=session)

    result = await delib.finalize(
        decision_id,
        description="I encountered an error processing your request. Please try again.",
        confidence=0.3,
        session=session,
    )
    assert result is None


async def test_finalize_passes_valid_decision(delib, brain, session):
    """009.5: Valid decisions pass the quality gate."""
    frame = _frame()
    decision_id = await delib.start("nous-default", "test valid decision", frame, session=session)

    result = await delib.finalize(
        decision_id,
        description="Final: Use PostgreSQL with pgvector for semantic search",
        confidence=0.85,
        session=session,
    )
    # Should return the decision_id (not None)
    assert result is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_deliberation.py -k "quality" -v`
Expected: FAIL — `finalize()` doesn't return None / doesn't have quality gate

**Step 3: Implement quality gate in `finalize()`**

Update `finalize()` in `nous/cognitive/deliberation.py`:

```python
# 009.5: Chat pattern prefixes that indicate noise
_QUALITY_CHAT_PREFIXES = (
    "done", "on it", "here's", "got it", "sure",
    "okay", "alright", "working on", "let me", "i'll",
)


def _validate_decision_quality(description: str, confidence: float, stakes: str) -> str | None:
    """Validate decision quality (009.5). Returns rejection reason or None if valid."""
    desc_stripped = description.strip()

    # Rule 1: Description too short
    if len(desc_stripped) < 20:
        return f"Description too short ({len(desc_stripped)} chars < 20)"

    # Rule 2: Default confidence + high stakes (never deliberated)
    if confidence == 0.5 and stakes in ("high", "critical"):
        return f"Default confidence (0.5) with {stakes} stakes"

    # Rule 3: Chat pattern prefix
    desc_lower = desc_stripped.lower()
    for prefix in _QUALITY_CHAT_PREFIXES:
        if desc_lower.startswith(prefix):
            return f"Description starts with chat pattern: '{prefix}'"

    # Rule 4: Error template
    if "encountered an error processing your request" in desc_lower:
        return "Description is an error message template"

    return None
```

Update `finalize()` to use it:

```python
async def finalize(
    self,
    decision_id: str,
    description: str,
    confidence: float,
    context: str | None = None,
    pattern: str | None = None,
    tags: list[str] | None = None,
    session: AsyncSession | None = None,
) -> str | None:
    """Update decision with final outcome. Returns decision_id or None if rejected (009.5).

    P1-3: Uses extended Brain.update() with confidence param.
    P2-6: Converts str decision_id to UUID for Brain.update().
    009.5: Validates quality before persisting. Deletes on rejection.
    """
    # 009.5: Fetch stakes from existing decision for quality gate
    detail = await self._brain.get(UUID(decision_id), session=session)
    if detail is None:
        return None

    stakes = detail.stakes

    # 009.5: Quality gate
    rejection = _validate_decision_quality(description, confidence, stakes)
    if rejection:
        logger.info("Quality gate rejected decision %s: %s", decision_id, rejection)
        await self.delete(decision_id, session=session)
        return None

    await self._brain.update(
        decision_id=UUID(decision_id),
        description=description,
        context=context,
        pattern=pattern,
        confidence=confidence,
        tags=tags,
        session=session,
    )
    return decision_id
```

**Step 4: Update `layer.py` to handle `None` return from `finalize()`**

In `nous/cognitive/layer.py`, the `post_turn()` method (around line 482-487) calls `finalize()` but doesn't use the return value. Since `finalize()` now handles its own deletion, no changes needed in `layer.py` — the None return is silently ignored.

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_deliberation.py -k "quality or valid" -v`
Expected: All pass

**Step 6: Run full test suites**

Run: `uv run pytest tests/test_deliberation.py tests/test_cognitive_layer.py -v`
Expected: All pass

**Step 7: Commit**

```bash
git add nous/cognitive/deliberation.py tests/test_deliberation.py
git commit -m "feat(009.5): add quality gate to deliberation finalize"
```

---

### Task 7: Integration Test — Full Pipeline

**Files:**
- Modify: `tests/test_cognitive_layer.py`

**Step 1: Write integration test**

Add to `tests/test_cognitive_layer.py`:

```python
# ---------------------------------------------------------------------------
# 009.5: Integration — full quality gate pipeline
# ---------------------------------------------------------------------------

async def test_task_frame_no_deliberation(cognitive_layer):
    """009.5: Task frame no longer triggers auto-deliberation."""
    # Simulate pre_turn with task frame
    # This should NOT start a deliberation
    frame = FrameSelection(
        frame_id="task",
        frame_name="Task Execution",
        confidence=0.9,
        match_method="pattern",
        default_category="tooling",
        default_stakes="medium",
    )
    result = await cognitive_layer._deliberation.should_deliberate(frame)
    assert result is False
```

**Step 2: Run test**

Run: `uv run pytest tests/test_cognitive_layer.py::test_task_frame_no_deliberation -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All pass. If any unrelated failures, investigate but don't fix in this task.

**Step 4: Commit**

```bash
git add tests/test_cognitive_layer.py
git commit -m "test(009.5): add integration test for quality gate pipeline"
```

---

### Task 8: Final Verification & Cleanup

**Files:**
- Review: all modified files

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All tests pass

**Step 2: Check for any import issues**

Run: `uv run python -c "from nous.cognitive.deliberation import DeliberationEngine; from nous.cognitive.layer import CognitiveLayer; print('OK')"`
Expected: `OK`

**Step 3: Review all changes**

Run: `git diff main --stat`
Verify only expected files changed.

**Step 4: Final commit if any cleanup needed**

Only if there are small fixes from the review.
