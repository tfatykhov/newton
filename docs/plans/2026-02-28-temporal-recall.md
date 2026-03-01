# 008.6 Temporal Recall Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add temporal recall to Nous so the agent always knows about recent conversations regardless of topic similarity.

**Architecture:** 3-tier escalation through context assembly. Tier 1: always-on episode titles (~50 tokens). Tier 2: boosted budget with summaries on recap detection. Tier 3: explicit `recall_recent` tool. Fix `list_recent()` bug as prerequisite.

**Tech Stack:** Python 3.12+, SQLAlchemy 2.0 async, pytest + pytest-asyncio

**Design doc:** `docs/plans/2026-02-28-temporal-recall-design.md`
**Spec:** `docs/implementation/008.6-temporal-recall.md`

---

## Phase 1: Fix `list_recent()` Bug

### Task 1: Fix `list_recent()` active filter + add `hours` param

**Files:**
- Modify: `nous/heart/episodes.py:311-354`
- Modify: `nous/heart/heart.py:128-135`
- Test: `tests/test_temporal_recall.py` (new file)

**Step 1: Write the failing tests**

Create `tests/test_temporal_recall.py`:

```python
"""Tests for 008.6 Temporal Recall."""

import pytest
from datetime import datetime, timedelta, UTC
from uuid import uuid4

from nous.heart.schemas import EpisodeSummary


# ---------------------------------------------------------------------------
# Phase 1: list_recent() fix
# ---------------------------------------------------------------------------

class TestListRecentFix:
    """list_recent() should return closed episodes, not ongoing ones."""

    @pytest.fixture
    def closed_episode(self):
        """An episode that has been properly ended (active=False, ended_at set)."""
        return {
            "agent_id": "test-agent",
            "title": "Ski Trip Planning",
            "summary": "Discussed budget for Breckenridge trip",
            "outcome": "success",
            "active": False,
            "started_at": datetime.now(UTC) - timedelta(hours=1),
            "ended_at": datetime.now(UTC) - timedelta(minutes=30),
        }

    @pytest.fixture
    def ongoing_episode(self):
        """An episode that is still active (active=True, no ended_at)."""
        return {
            "agent_id": "test-agent",
            "title": "Current Chat",
            "summary": "Ongoing conversation",
            "outcome": None,
            "active": True,
            "started_at": datetime.now(UTC) - timedelta(minutes=10),
            "ended_at": None,
        }

    @pytest.mark.asyncio
    async def test_list_recent_returns_closed_episodes(self, db, closed_episode):
        """After 008.3, closed episodes have active=False. list_recent must include them."""
        from nous.storage.models import Episode as EpisodeORM
        from nous.heart.episodes import EpisodeManager

        async with db.session() as session:
            ep = EpisodeORM(id=uuid4(), **closed_episode)
            session.add(ep)
            await session.flush()

            mgr = EpisodeManager(db, "test-agent")
            results = await mgr.list_recent(limit=10, session=session)

            assert len(results) == 1
            assert results[0].title == "Ski Trip Planning"

    @pytest.mark.asyncio
    async def test_list_recent_excludes_ongoing_episodes(self, db, ongoing_episode):
        """Ongoing episodes (ended_at=None) should NOT appear in list_recent."""
        from nous.storage.models import Episode as EpisodeORM
        from nous.heart.episodes import EpisodeManager

        async with db.session() as session:
            ep = EpisodeORM(id=uuid4(), **ongoing_episode)
            session.add(ep)
            await session.flush()

            mgr = EpisodeManager(db, "test-agent")
            results = await mgr.list_recent(limit=10, session=session)

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_list_recent_hours_filter(self, db, closed_episode):
        """hours param should filter to episodes within time window."""
        from nous.storage.models import Episode as EpisodeORM
        from nous.heart.episodes import EpisodeManager

        async with db.session() as session:
            # Recent episode (1 hour ago)
            ep1 = EpisodeORM(id=uuid4(), **closed_episode)
            session.add(ep1)

            # Old episode (3 days ago)
            old = closed_episode.copy()
            old["title"] = "Old Episode"
            old["started_at"] = datetime.now(UTC) - timedelta(days=3)
            old["ended_at"] = datetime.now(UTC) - timedelta(days=3) + timedelta(minutes=30)
            ep2 = EpisodeORM(id=uuid4(), **old)
            session.add(ep2)
            await session.flush()

            mgr = EpisodeManager(db, "test-agent")

            # 48 hours: only recent
            results = await mgr.list_recent(limit=10, hours=48, session=session)
            assert len(results) == 1
            assert results[0].title == "Ski Trip Planning"

            # 100 hours: both
            results = await mgr.list_recent(limit=10, hours=100, session=session)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_recent_ordered_by_started_at_desc(self, db):
        """Results should be newest first."""
        from nous.storage.models import Episode as EpisodeORM
        from nous.heart.episodes import EpisodeManager

        async with db.session() as session:
            now = datetime.now(UTC)
            for i, title in enumerate(["Third", "First", "Second"]):
                ep = EpisodeORM(
                    id=uuid4(),
                    agent_id="test-agent",
                    title=title,
                    summary=f"Episode {title}",
                    outcome="success",
                    active=False,
                    started_at=now - timedelta(hours=[3, 1, 2][i]),
                    ended_at=now - timedelta(hours=[3, 1, 2][i]) + timedelta(minutes=30),
                )
                session.add(ep)
            await session.flush()

            mgr = EpisodeManager(db, "test-agent")
            results = await mgr.list_recent(limit=10, session=session)

            assert [r.title for r in results] == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_list_recent_excludes_abandoned(self, db):
        """Abandoned episodes should be excluded by default."""
        from nous.storage.models import Episode as EpisodeORM
        from nous.heart.episodes import EpisodeManager

        async with db.session() as session:
            ep = EpisodeORM(
                id=uuid4(),
                agent_id="test-agent",
                title="Abandoned",
                summary="Abandoned episode",
                outcome="abandoned",
                active=False,
                started_at=datetime.now(UTC) - timedelta(hours=1),
                ended_at=datetime.now(UTC) - timedelta(minutes=30),
            )
            session.add(ep)
            await session.flush()

            mgr = EpisodeManager(db, "test-agent")
            results = await mgr.list_recent(limit=10, session=session)

            assert len(results) == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal_recall.py -v -x`
Expected: FAIL — `list_recent()` filters `active=True` so closed episodes are excluded.

**Step 3: Fix `list_recent()` implementation**

In `nous/heart/episodes.py`, modify the `list_recent()` public method signature and `_list_recent()` implementation:

```python
# Public method — add hours param
async def list_recent(
    self,
    limit: int = 10,
    outcome: str | None = None,
    session: AsyncSession | None = None,
    hours: int | None = None,
) -> list[EpisodeSummary]:
    """List recent episodes ordered by started_at DESC."""
    if session is None:
        async with self.db.session() as session:
            return await self._list_recent(limit, outcome, session, hours)
    return await self._list_recent(limit, outcome, session, hours)

# Private method — fix filter + add hours
async def _list_recent(
    self,
    limit: int,
    outcome: str | None,
    session: AsyncSession,
    hours: int | None = None,
) -> list[EpisodeSummary]:
    stmt = (
        select(Episode)
        .where(Episode.agent_id == self.agent_id)
        .where(Episode.ended_at.isnot(None))  # 008.6: Completed episodes only
        .order_by(Episode.started_at.desc())
        .limit(limit)
    )
    if hours is not None:
        from datetime import datetime, timedelta, UTC
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        stmt = stmt.where(Episode.started_at >= cutoff)
    if outcome is not None:
        stmt = stmt.where(Episode.outcome == outcome)
    else:
        stmt = stmt.where(Episode.outcome != 'abandoned')

    result = await session.execute(stmt)
    episodes = result.scalars().all()

    return [
        EpisodeSummary(
            id=e.id,
            title=e.title,
            summary=e.summary,
            outcome=e.outcome,
            started_at=e.started_at,
            tags=e.tags or [],
        )
        for e in episodes
    ]
```

Also update `Heart.list_episodes()` in `nous/heart/heart.py` to pass through the `hours` param:

```python
async def list_episodes(
    self,
    limit: int = 10,
    outcome: str | None = None,
    session: AsyncSession | None = None,
    hours: int | None = None,
) -> list[EpisodeSummary]:
    """List recent episodes."""
    return await self.episodes.list_recent(limit, outcome, session, hours)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_temporal_recall.py::TestListRecentFix -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add nous/heart/episodes.py nous/heart/heart.py tests/test_temporal_recall.py
git commit -m "fix(008.6): list_recent returns closed episodes, add hours param

After 008.3, closed episodes have active=False. Change filter from
active=True to ended_at IS NOT NULL. Add optional hours parameter
for time-windowed queries."
```

---

## Phase 2: Always-On Temporal Context Tier

### Task 2: Add config toggle

**Files:**
- Modify: `nous/config.py:56-64`
- Test: `tests/test_temporal_recall.py`

**Step 1: Write the failing test**

Add to `tests/test_temporal_recall.py`:

```python
# ---------------------------------------------------------------------------
# Phase 2: Config
# ---------------------------------------------------------------------------

class TestTemporalConfig:
    """NOUS_TEMPORAL_CONTEXT_ENABLED config toggle."""

    def test_temporal_context_enabled_default(self):
        """Default should be True."""
        from nous.config import Settings
        s = Settings(anthropic_api_key="test")
        assert s.temporal_context_enabled is True

    def test_temporal_context_disabled(self, monkeypatch):
        """Can be disabled via env var."""
        from nous.config import Settings
        monkeypatch.setenv("NOUS_TEMPORAL_CONTEXT_ENABLED", "false")
        s = Settings(anthropic_api_key="test")
        assert s.temporal_context_enabled is False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal_recall.py::TestTemporalConfig -v`
Expected: FAIL — `temporal_context_enabled` attribute doesn't exist yet.

**Step 3: Add config field**

In `nous/config.py`, add after the `decision_review_enabled` line (around line 60):

```python
    temporal_context_enabled: bool = True
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_temporal_recall.py::TestTemporalConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/config.py tests/test_temporal_recall.py
git commit -m "feat(008.6): add NOUS_TEMPORAL_CONTEXT_ENABLED config toggle"
```

### Task 3: Add temporal tier to context assembly

**Files:**
- Modify: `nous/cognitive/context.py:362-400`
- Test: `tests/test_temporal_recall.py`

**Step 1: Write the failing tests**

Add to `tests/test_temporal_recall.py`:

```python
# ---------------------------------------------------------------------------
# Phase 2: Temporal context tier
# ---------------------------------------------------------------------------

class TestTemporalContextTier:
    """Always-on temporal tier injects recent episode titles into context."""

    @pytest.fixture
    def mock_heart(self):
        """Heart with controllable list_episodes."""
        from unittest.mock import AsyncMock, MagicMock
        heart = MagicMock()
        heart.list_episodes = AsyncMock(return_value=[])
        heart.search_episodes = AsyncMock(return_value=[])
        heart.search_facts = AsyncMock(return_value=[])
        heart.search_procedures = AsyncMock(return_value=[])
        heart.search_censors = AsyncMock(return_value=[])
        heart.get_facts_by_categories = AsyncMock(return_value=[])
        heart.get_working_memory = AsyncMock(return_value=None)
        return heart

    @pytest.fixture
    def mock_brain(self):
        from unittest.mock import AsyncMock, MagicMock
        brain = MagicMock()
        brain.query = AsyncMock(return_value=[])
        brain.embeddings = None  # No embeddings for unit tests
        return brain

    @pytest.fixture
    def settings(self):
        from nous.config import Settings
        return Settings(anthropic_api_key="test", temporal_context_enabled=True)

    @pytest.fixture
    def frame(self):
        from nous.cognitive.schemas import FrameSelection
        return FrameSelection(
            frame_id="task",
            frame_name="Task",
            confidence=1.0,
            match_method="pattern",
        )

    @pytest.mark.asyncio
    async def test_temporal_tier_includes_recent_titles(self, mock_heart, mock_brain, settings, frame):
        """Recent episode titles should appear in context when temporal_context_enabled=True."""
        from nous.cognitive.context import ContextEngine
        from nous.heart.schemas import EpisodeSummary

        episodes = [
            EpisodeSummary(
                id=uuid4(), title="Ski Trip Planning",
                summary="Budget discussion", outcome="success",
                started_at=datetime.now(UTC) - timedelta(hours=1), tags=["travel"],
            ),
            EpisodeSummary(
                id=uuid4(), title="Code Review",
                summary="Reviewed PR #81", outcome="success",
                started_at=datetime.now(UTC) - timedelta(hours=3), tags=["dev"],
            ),
        ]
        mock_heart.list_episodes = AsyncMock(return_value=episodes)

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build("test-agent", "session-1", "hello", frame)

        # Should contain a section with "Recent conversations"
        section_labels = [s.label for s in result.sections]
        assert "Recent Conversations" in section_labels

        recent_section = next(s for s in result.sections if s.label == "Recent Conversations")
        assert "Ski Trip Planning" in recent_section.content
        assert "Code Review" in recent_section.content

    @pytest.mark.asyncio
    async def test_temporal_tier_disabled_by_config(self, mock_heart, mock_brain, frame):
        """When temporal_context_enabled=False, no temporal tier should appear."""
        from nous.cognitive.context import ContextEngine
        from nous.heart.schemas import EpisodeSummary

        settings = Settings(anthropic_api_key="test", temporal_context_enabled=False)
        episodes = [
            EpisodeSummary(
                id=uuid4(), title="Ski Trip",
                summary="Trip stuff", outcome="success",
                started_at=datetime.now(UTC) - timedelta(hours=1), tags=[],
            ),
        ]
        mock_heart.list_episodes = AsyncMock(return_value=episodes)

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build("test-agent", "session-1", "hello", frame)

        section_labels = [s.label for s in result.sections]
        assert "Recent Conversations" not in section_labels

    @pytest.mark.asyncio
    async def test_temporal_tier_deduplicates_with_semantic(self, mock_heart, mock_brain, settings, frame):
        """Episodes shown in temporal tier should be excluded from semantic results."""
        from nous.cognitive.context import ContextEngine
        from nous.heart.schemas import EpisodeSummary

        shared_id = uuid4()
        temporal_ep = EpisodeSummary(
            id=shared_id, title="Ski Trip",
            summary="Budget discussion", outcome="success",
            started_at=datetime.now(UTC) - timedelta(hours=1), tags=["travel"],
        )
        semantic_ep = EpisodeSummary(
            id=shared_id, title="Ski Trip",
            summary="Budget discussion", outcome="success",
            started_at=datetime.now(UTC) - timedelta(hours=1), tags=["travel"],
            score=0.85,
        )

        mock_heart.list_episodes = AsyncMock(return_value=[temporal_ep])
        mock_heart.search_episodes = AsyncMock(return_value=[semantic_ep])

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build("test-agent", "session-1", "ski trip", frame)

        # "Ski Trip" should appear in temporal tier but NOT duplicated in Past Episodes
        ski_mentions = result.system_prompt.count("Ski Trip")
        assert ski_mentions == 1  # Only in temporal tier, not in semantic episodes too

    @pytest.mark.asyncio
    async def test_temporal_tier_empty_when_no_recent(self, mock_heart, mock_brain, settings, frame):
        """No temporal section when no recent episodes exist."""
        from nous.cognitive.context import ContextEngine

        mock_heart.list_episodes = AsyncMock(return_value=[])

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build("test-agent", "session-1", "hello", frame)

        section_labels = [s.label for s in result.sections]
        assert "Recent Conversations" not in section_labels
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal_recall.py::TestTemporalContextTier -v`
Expected: FAIL — no temporal tier logic exists yet.

**Step 3: Implement temporal tier in context assembly**

In `nous/cognitive/context.py`:

1. Add import at top:
```python
from datetime import datetime, timedelta, UTC
```

2. In `build()`, **before** the `# 8. Episodes` block (before line 362), add the temporal tier:

```python
        # 7.5 Temporal awareness — always include recent episode titles (008.6)
        _temporal_episode_ids: set[str] = set()
        if self._settings.temporal_context_enabled and budget.episodes > 0:
            try:
                recent = await self._heart.list_episodes(limit=5, hours=48)
                if recent:
                    _temporal_episode_ids = {str(e.id) for e in recent}
                    recent_lines = []
                    for e in recent:
                        title = e.title or (e.summary[:60] if e.summary else "Untitled")
                        time_str = e.started_at.strftime("%b %d %H:%M")
                        recent_lines.append(f"- [{time_str}] {title}")
                    recent_text = "\n".join(recent_lines)
                    sections.append(
                        ContextSection(
                            priority=7,  # Between procedures (7) and episodes (8)
                            label="Recent Conversations",
                            content=recent_text,
                            token_estimate=self._estimate_tokens(recent_text),
                        )
                    )
            except Exception as e:
                logger.warning("Temporal tier failed: %s", e)
```

3. In the `# 8. Episodes` block, filter out temporal episode IDs to avoid duplicates. After `episodes = await self._heart.search_episodes(...)` and before the threshold check, add:

```python
                # 008.6: Exclude episodes already shown in temporal tier
                if _temporal_episode_ids:
                    episodes = [e for e in episodes if str(e.id) not in _temporal_episode_ids]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_temporal_recall.py::TestTemporalContextTier -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add nous/cognitive/context.py tests/test_temporal_recall.py
git commit -m "feat(008.6): always-on temporal context tier

Inject recent episode titles (last 48h, up to 5) into every context
window at priority 7.5. ~50 tokens cost. Deduplicates with semantic
episode search by ID. Gated by temporal_context_enabled config."
```

---

## Phase 3: `recall_recent` Tool

### Task 4: Implement `recall_recent` tool + register in all frames

**Files:**
- Modify: `nous/api/tools.py:270-335, 544-555`
- Modify: `nous/api/runner.py:38-46`
- Test: `tests/test_temporal_recall.py`

**Step 1: Write the failing tests**

Add to `tests/test_temporal_recall.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Phase 3: recall_recent tool
# ---------------------------------------------------------------------------

class TestRecallRecentTool:
    """recall_recent tool returns time-ordered episodes."""

    @pytest.fixture
    def mock_heart(self):
        from unittest.mock import AsyncMock, MagicMock
        heart = MagicMock()
        heart.list_episodes = AsyncMock(return_value=[])
        return heart

    @pytest.mark.asyncio
    async def test_recall_recent_returns_formatted_episodes(self, mock_heart):
        """recall_recent should return formatted list of recent episodes."""
        from nous.heart.schemas import EpisodeSummary
        from nous.api.tools import create_nous_tools
        from nous.brain.brain import Brain

        mock_brain = MagicMock(spec=Brain)
        episodes = [
            EpisodeSummary(
                id=uuid4(), title="Ski Trip Planning",
                summary="Budget for Breckenridge", outcome="success",
                started_at=datetime(2026, 2, 28, 12, 24, tzinfo=UTC), tags=["travel"],
            ),
            EpisodeSummary(
                id=uuid4(), title="Code Review",
                summary="Reviewed PR #81", outcome="success",
                started_at=datetime(2026, 2, 28, 11, 0, tzinfo=UTC), tags=["dev"],
            ),
        ]
        mock_heart.list_episodes = AsyncMock(return_value=episodes)

        tools = create_nous_tools(mock_brain, mock_heart)
        result = await tools["recall_recent"](hours=48, limit=10)

        text = result["content"][0]["text"]
        assert "Ski Trip Planning" in text
        assert "Code Review" in text
        assert "Feb 28" in text

    @pytest.mark.asyncio
    async def test_recall_recent_empty(self, mock_heart):
        """recall_recent with no episodes returns informative message."""
        from nous.api.tools import create_nous_tools
        from nous.brain.brain import Brain

        mock_brain = MagicMock(spec=Brain)
        mock_heart.list_episodes = AsyncMock(return_value=[])

        tools = create_nous_tools(mock_brain, mock_heart)
        result = await tools["recall_recent"](hours=48, limit=10)

        text = result["content"][0]["text"]
        assert "No episodes" in text

    @pytest.mark.asyncio
    async def test_recall_recent_passes_hours_and_limit(self, mock_heart):
        """hours and limit params should be forwarded to list_episodes."""
        from nous.api.tools import create_nous_tools
        from nous.brain.brain import Brain

        mock_brain = MagicMock(spec=Brain)
        tools = create_nous_tools(mock_brain, mock_heart)
        await tools["recall_recent"](hours=72, limit=5)

        mock_heart.list_episodes.assert_called_once_with(limit=5, hours=72)

    def test_recall_recent_in_frame_tools(self):
        """recall_recent should be available in all non-initiation frames."""
        from nous.api.runner import FRAME_TOOLS

        for frame, tools in FRAME_TOOLS.items():
            if frame == "initiation":
                assert "recall_recent" not in tools
            elif tools == ["*"]:
                pass  # task frame includes all tools
            else:
                assert "recall_recent" in tools, f"recall_recent missing from {frame} frame"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal_recall.py::TestRecallRecentTool -v`
Expected: FAIL — `recall_recent` doesn't exist yet.

**Step 3: Implement recall_recent tool**

In `nous/api/tools.py`:

1. Add the `recall_recent` function inside `create_nous_tools()` (after `recall_deep`, around line 335):

```python
    async def recall_recent(
        hours: int = 48,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Recall recent episodes by time, not topic similarity.

        Use this when the user asks "what did we talk about", "what happened
        recently", or you need a comprehensive overview of recent activity.
        Semantic recall (recall_deep) matches by meaning — this matches by time.

        Args:
            hours: Look back this many hours (default 48)
            limit: Maximum episodes to return (default 10)

        Returns:
            MCP-compliant response with time-ordered episode list
        """
        try:
            episodes = await heart.list_episodes(limit=limit, hours=hours)

            if not episodes:
                return {"content": [{"type": "text", "text": f"No episodes found in the last {hours} hours."}]}

            lines = [f"Recent episodes (last {hours}h):"]
            for e in episodes:
                title = e.title or (e.summary[:60] if e.summary else "Untitled")
                time_str = e.started_at.strftime("%b %d %H:%M")
                lines.append(f"- [{time_str}] {title}")
                if e.summary and e.summary != e.title:
                    lines.append(f"  {e.summary[:150]}")

            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        except Exception as e:
            logger.exception("recall_recent tool failed")
            return {"content": [{"type": "text", "text": f"Error fetching recent episodes: {e}"}]}
```

2. Add the tool schema constant (after `_RECALL_DEEP_SCHEMA`):

```python
_RECALL_RECENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Recall recent episodes by time (not topic similarity). Use when the user asks what you discussed recently or you need a temporal overview.",
    "properties": {
        "hours": {
            "type": "integer",
            "description": "Look back this many hours (default 48)",
            "default": 48,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum episodes to return (default 10)",
            "default": 10,
        },
    },
    "required": [],
}
```

3. Add to `create_nous_tools()` return dict — include `"recall_recent": recall_recent` in the returned dict.

4. In `register_nous_tools()`, add:
```python
    dispatcher.register("recall_recent", closures["recall_recent"], _RECALL_RECENT_SCHEMA)
```

5. In `nous/api/runner.py`, add `"recall_recent"` to every frame list in `FRAME_TOOLS` (except `initiation`):

```python
FRAME_TOOLS: dict[str, list[str]] = {
    "conversation": ["record_decision", "learn_fact", "recall_deep", "recall_recent", "create_censor", "bash", "read_file", "write_file", "web_search", "web_fetch"],
    "question": ["recall_deep", "recall_recent", "bash", "read_file", "write_file", "record_decision", "learn_fact", "create_censor", "web_search", "web_fetch"],
    "decision": ["record_decision", "recall_deep", "recall_recent", "create_censor", "bash", "read_file", "web_search", "web_fetch"],
    "creative": ["learn_fact", "recall_deep", "recall_recent", "write_file", "web_search"],
    "task": ["*"],  # All tools
    "debug": ["record_decision", "recall_deep", "recall_recent", "bash", "read_file", "learn_fact", "web_search", "web_fetch"],
    "initiation": ["store_identity", "complete_initiation"],
}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_temporal_recall.py::TestRecallRecentTool -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add nous/api/tools.py nous/api/runner.py tests/test_temporal_recall.py
git commit -m "feat(008.6): add recall_recent tool for time-based recall

New agent tool returns episodes ordered by time, not similarity.
Available in all frames except initiation. Takes hours (default 48)
and limit (default 10) parameters."
```

---

## Phase 4: Recap Detection + Budget Boost

### Task 5: Add recap detection to cognitive layer

**Files:**
- Modify: `nous/cognitive/layer.py`
- Test: `tests/test_temporal_recall.py`

**Step 1: Write the failing tests**

Add to `tests/test_temporal_recall.py`:

```python
# ---------------------------------------------------------------------------
# Phase 4: Recap detection
# ---------------------------------------------------------------------------

class TestRecapDetection:
    """Recap pattern detection in cognitive layer."""

    @pytest.fixture
    def layer_cls(self):
        """Import CognitiveLayer to test _is_recap_query."""
        from nous.cognitive.layer import CognitiveLayer
        return CognitiveLayer

    @pytest.mark.parametrize("query", [
        "what did we talk about",
        "What did we talk about recently?",
        "what have we discussed",
        "catch me up",
        "recap",
        "give me a recap of recent conversations",
        "what happened today",
        "summary of recent discussions",
        "what did we do yesterday",
    ])
    def test_recap_patterns_detected(self, query):
        """Known recap patterns should be detected."""
        from nous.cognitive.layer import _is_recap_query
        assert _is_recap_query(query) is True

    @pytest.mark.parametrize("query", [
        "how do I fix this bug",
        "what is the capital of France",
        "write a function to sort a list",
        "hello",
        "thanks",
        "tell me about the architecture",
    ])
    def test_non_recap_not_detected(self, query):
        """Normal queries should NOT trigger recap detection."""
        from nous.cognitive.layer import _is_recap_query
        assert _is_recap_query(query) is False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal_recall.py::TestRecapDetection -v`
Expected: FAIL — `_is_recap_query` doesn't exist yet.

**Step 3: Implement recap detection**

In `nous/cognitive/layer.py`, add near the top (after imports, before the class):

```python
# 008.6: Recap query detection
_RECAP_PATTERNS = frozenset({
    "what did we talk about",
    "what have we discussed",
    "what did we do",
    "recent conversations",
    "catch me up",
    "what happened",
    "recap",
    "summary of recent",
})


def _is_recap_query(user_input: str) -> bool:
    """Detect if user is asking for a temporal recap."""
    lower = user_input.lower().strip()
    return any(p in lower for p in _RECAP_PATTERNS)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_temporal_recall.py::TestRecapDetection -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add nous/cognitive/layer.py tests/test_temporal_recall.py
git commit -m "feat(008.6): recap query detection patterns

Module-level _is_recap_query() function with conservative pattern set.
Detects 'what did we talk about', 'catch me up', 'recap', etc."
```

### Task 6: Wire recap detection + temporal_recency into budget boost

**Files:**
- Modify: `nous/cognitive/layer.py`
- Modify: `nous/cognitive/context.py`
- Modify: `nous/cognitive/intent.py`
- Test: `tests/test_temporal_recall.py`

**Step 1: Write the failing tests**

Add to `tests/test_temporal_recall.py`:

```python
class TestTemporalRecencyWiring:
    """temporal_recency signals from intent classifier boost episode budget."""

    def test_plan_retrieval_boosts_episodes_on_high_recency(self):
        """When temporal_recency > 0.5, episode budget should be boosted."""
        from nous.cognitive.intent import IntentClassifier, IntentSignals
        from nous.cognitive.schemas import FrameSelection

        classifier = IntentClassifier()
        frame = FrameSelection(
            frame_id="conversation",
            frame_name="Conversation",
            confidence=1.0,
            match_method="pattern",
        )

        signals = classifier.classify("what did we talk about recently", frame)
        assert signals.temporal_recency > 0.5

        plan = classifier.plan_retrieval(signals, input_text="what did we talk about recently")
        # Episode budget should not be 0 (conversation frame default is 0)
        assert plan.budget_overrides.get("episodes", 0) > 0


class TestBudgetBoost:
    """Context assembly boosts episode budget when temporal_boost=True."""

    @pytest.fixture
    def mock_heart(self):
        from unittest.mock import AsyncMock, MagicMock
        heart = MagicMock()
        heart.list_episodes = AsyncMock(return_value=[])
        heart.search_episodes = AsyncMock(return_value=[])
        heart.search_facts = AsyncMock(return_value=[])
        heart.search_procedures = AsyncMock(return_value=[])
        heart.search_censors = AsyncMock(return_value=[])
        heart.get_facts_by_categories = AsyncMock(return_value=[])
        heart.get_working_memory = AsyncMock(return_value=None)
        return heart

    @pytest.fixture
    def mock_brain(self):
        from unittest.mock import AsyncMock, MagicMock
        brain = MagicMock()
        brain.query = AsyncMock(return_value=[])
        brain.embeddings = None
        return brain

    @pytest.fixture
    def settings(self):
        from nous.config import Settings
        return Settings(anthropic_api_key="test", temporal_context_enabled=True)

    @pytest.fixture
    def frame(self):
        from nous.cognitive.schemas import FrameSelection
        return FrameSelection(
            frame_id="conversation",
            frame_name="Conversation",
            confidence=1.0,
            match_method="pattern",
        )

    @pytest.mark.asyncio
    async def test_temporal_boost_includes_summaries(self, mock_heart, mock_brain, settings, frame):
        """When temporal_boost=True, temporal tier should include summaries, not just titles."""
        from nous.cognitive.context import ContextEngine
        from nous.heart.schemas import EpisodeSummary

        episodes = [
            EpisodeSummary(
                id=uuid4(), title="Ski Trip Planning",
                summary="Discussed budget and dates for Breckenridge ski trip in March 2026",
                outcome="success",
                started_at=datetime.now(UTC) - timedelta(hours=1), tags=["travel"],
            ),
        ]
        mock_heart.list_episodes = AsyncMock(return_value=episodes)

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build(
            "test-agent", "session-1", "what did we talk about", frame,
            temporal_boost=True,
        )

        section_labels = [s.label for s in result.sections]
        assert "Recent Conversations" in section_labels

        recent_section = next(s for s in result.sections if s.label == "Recent Conversations")
        # With boost, summaries should be included (not just titles)
        assert "Discussed budget" in recent_section.content
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal_recall.py::TestTemporalRecencyWiring -v`
Run: `uv run pytest tests/test_temporal_recall.py::TestBudgetBoost -v`
Expected: FAIL — no wiring, no temporal_boost param.

**Step 3: Wire recap detection + budget boost**

**3a.** In `nous/cognitive/intent.py`, modify `plan_retrieval()` to boost episodes when temporal_recency is high. After the frame-based budget overrides block (around line 205), add:

```python
        # 008.6: Temporal recency boost — ensure episodes are retrieved
        if signals.temporal_recency > 0.5:
            current_ep_budget = plan.budget_overrides.get("episodes", None)
            if current_ep_budget is not None and current_ep_budget == 0:
                plan.budget_overrides["episodes"] = 1000  # Override zero to meaningful budget
            # Boost episode query limit
            for q in plan.queries:
                if q.memory_type == "episode":
                    q.limit = max(q.limit, 8)
```

**3b.** In `nous/cognitive/context.py`, modify `build()` signature to accept `temporal_boost`:

```python
    async def build(
        self,
        agent_id: str,
        session_id: str,
        input_text: str,
        frame: FrameSelection,
        session: AsyncSession | None = None,
        *,
        conversation_messages: list[str] | None = None,
        retrieval_plan: RetrievalPlan | None = None,
        usage_tracker: UsageTracker | None = None,
        identity_override: str | None = None,
        temporal_boost: bool = False,  # 008.6: Include summaries in temporal tier
    ) -> BuildResult:
```

Then modify the temporal tier block to include summaries when `temporal_boost=True`:

```python
        # 7.5 Temporal awareness (008.6)
        _temporal_episode_ids: set[str] = set()
        if self._settings.temporal_context_enabled and budget.episodes > 0:
            try:
                recent = await self._heart.list_episodes(limit=5, hours=48)
                if recent:
                    _temporal_episode_ids = {str(e.id) for e in recent}
                    recent_lines = []
                    for e in recent:
                        title = e.title or (e.summary[:60] if e.summary else "Untitled")
                        time_str = e.started_at.strftime("%b %d %H:%M")
                        recent_lines.append(f"- [{time_str}] {title}")
                        # 008.6: Include summaries when temporal boost is active
                        if temporal_boost and e.summary and e.summary != e.title:
                            recent_lines.append(f"  {e.summary[:200]}")
                    recent_text = "\n".join(recent_lines)
                    sections.append(
                        ContextSection(
                            priority=7,
                            label="Recent Conversations",
                            content=recent_text,
                            token_estimate=self._estimate_tokens(recent_text),
                        )
                    )
            except Exception as e:
                logger.warning("Temporal tier failed: %s", e)
```

**3c.** In `nous/cognitive/layer.py`, in `pre_turn()`, detect recap and pass `temporal_boost` to context build. After intent classification (around line 200), add:

```python
        # 008.6: Detect recap queries and set temporal boost
        _temporal_boost = _is_recap_query(user_input) or signals.temporal_recency > 0.5
```

Then pass it to `build()` call (around line 237):

```python
                build_result = await self._context.build(
                    agent_id,
                    session_id,
                    user_input,
                    frame,
                    session=session,
                    conversation_messages=conversation_messages,
                    retrieval_plan=plan,
                    usage_tracker=self._usage_tracker,
                    identity_override=_identity_override,
                    temporal_boost=_temporal_boost,  # 008.6
                )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_temporal_recall.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add nous/cognitive/layer.py nous/cognitive/context.py nous/cognitive/intent.py tests/test_temporal_recall.py
git commit -m "feat(008.6): recap detection + temporal budget boost

Wire recap patterns and temporal_recency signals to context assembly.
When detected: episode budget boosted (conversation frame 0→1000),
temporal tier includes summaries (not just titles), episode query
limit increased to 8. 3-tier escalation complete."
```

### Task 7: Run full test suite

**Step 1: Run all temporal recall tests**

Run: `uv run pytest tests/test_temporal_recall.py -v`
Expected: All tests pass (estimated ~20 tests).

**Step 2: Run full test suite for regressions**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: No new failures beyond pre-existing ones (25 known failures on main).

**Step 3: Commit any fixes if needed**

---

## Summary

| Phase | Tasks | Tests | Files Modified |
|-------|-------|-------|---------------|
| 1: Fix list_recent | 1 | 5 | episodes.py, heart.py |
| 2: Temporal tier | 2-3 | 6 | config.py, context.py |
| 3: recall_recent tool | 4 | 4 | tools.py, runner.py |
| 4: Recap + boost | 5-7 | ~7 | layer.py, context.py, intent.py |
| **Total** | **7 tasks** | **~22 tests** | **7 files** |
