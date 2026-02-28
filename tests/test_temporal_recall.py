"""Tests for temporal recall — spec 008.6 Phases 1-2.

Covers the list_recent() fix (active filter → ended_at filter),
the new `hours` parameter for time-windowed episode listing,
config toggle, and temporal context tier.

5 tests in TestListRecentFix, 2 in TestTemporalConfig, 4 in TestTemporalContextTier.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from nous.heart.schemas import EpisodeSummary
from nous.storage.models import Episode as EpisodeORM

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_ID = "nous-default"


def _make_episode(
    *,
    active: bool = False,
    ended_at: datetime | None = None,
    started_at: datetime | None = None,
    outcome: str | None = "success",
    summary: str = "test episode",
    title: str = "Test",
) -> EpisodeORM:
    """Build an Episode ORM instance with sensible defaults."""
    now = datetime.now(UTC)
    return EpisodeORM(
        id=uuid4(),
        agent_id=AGENT_ID,
        title=title,
        summary=summary,
        active=active,
        started_at=started_at or now,
        ended_at=ended_at,
        outcome=outcome,
        tags=["test"],
    )


# ---------------------------------------------------------------------------
# TestListRecentFix
# ---------------------------------------------------------------------------


class TestListRecentFix:
    """Tests for the fixed list_recent() behavior."""

    async def test_list_recent_returns_closed_episodes(self, heart, session):
        """Closed episodes (active=False, ended_at set) should be returned."""
        now = datetime.now(UTC)
        ep = _make_episode(
            active=False,
            ended_at=now,
            started_at=now - timedelta(minutes=10),
            outcome="success",
        )
        session.add(ep)
        await session.flush()

        results = await heart.list_episodes(limit=10, session=session)

        ids = [r.id for r in results]
        assert ep.id in ids

    async def test_list_recent_excludes_ongoing_episodes(self, heart, session):
        """Ongoing episodes (active=True, ended_at=None) should be excluded."""
        now = datetime.now(UTC)
        ep = _make_episode(
            active=True,
            ended_at=None,
            started_at=now - timedelta(minutes=5),
            outcome=None,
        )
        session.add(ep)
        await session.flush()

        results = await heart.list_episodes(limit=10, session=session)

        ids = [r.id for r in results]
        assert ep.id not in ids

    async def test_list_recent_hours_filter(self, heart, session):
        """hours parameter filters episodes by started_at time window."""
        now = datetime.now(UTC)

        recent_ep = _make_episode(
            active=False,
            ended_at=now - timedelta(minutes=30),
            started_at=now - timedelta(hours=1),
            outcome="success",
            summary="recent episode",
        )
        old_ep = _make_episode(
            active=False,
            ended_at=now - timedelta(days=2, hours=23),
            started_at=now - timedelta(days=3),
            outcome="success",
            summary="old episode",
        )
        session.add(recent_ep)
        session.add(old_ep)
        await session.flush()

        # hours=48 should return only the recent one
        results_48 = await heart.list_episodes(limit=10, hours=48, session=session)
        ids_48 = [r.id for r in results_48]
        assert recent_ep.id in ids_48
        assert old_ep.id not in ids_48

        # hours=100 should return both
        results_100 = await heart.list_episodes(limit=10, hours=100, session=session)
        ids_100 = [r.id for r in results_100]
        assert recent_ep.id in ids_100
        assert old_ep.id in ids_100

    async def test_list_recent_ordered_by_started_at_desc(self, heart, session):
        """Episodes should be returned newest-first."""
        now = datetime.now(UTC)

        ep1 = _make_episode(
            active=False,
            ended_at=now - timedelta(hours=2),
            started_at=now - timedelta(hours=3),
            outcome="success",
            summary="oldest",
        )
        ep2 = _make_episode(
            active=False,
            ended_at=now - timedelta(hours=1),
            started_at=now - timedelta(hours=2),
            outcome="partial",
            summary="middle",
        )
        ep3 = _make_episode(
            active=False,
            ended_at=now - timedelta(minutes=10),
            started_at=now - timedelta(hours=1),
            outcome="success",
            summary="newest",
        )
        session.add(ep1)
        session.add(ep2)
        session.add(ep3)
        await session.flush()

        results = await heart.list_episodes(limit=10, session=session)

        # Filter to only our test episodes
        our_ids = {ep1.id, ep2.id, ep3.id}
        our_results = [r for r in results if r.id in our_ids]

        assert len(our_results) == 3
        assert our_results[0].id == ep3.id  # newest first
        assert our_results[1].id == ep2.id
        assert our_results[2].id == ep1.id

    async def test_list_recent_excludes_abandoned(self, heart, session):
        """Abandoned episodes should be excluded by default."""
        now = datetime.now(UTC)
        ep = _make_episode(
            active=False,
            ended_at=now,
            started_at=now - timedelta(minutes=10),
            outcome="abandoned",
        )
        session.add(ep)
        await session.flush()

        results = await heart.list_episodes(limit=10, session=session)

        ids = [r.id for r in results]
        assert ep.id not in ids


# ---------------------------------------------------------------------------
# TestTemporalConfig (Phase 2, Task 2)
# ---------------------------------------------------------------------------


class TestTemporalConfig:
    """Tests for the temporal_context_enabled config toggle."""

    def test_temporal_context_enabled_default(self):
        from nous.config import Settings

        s = Settings(**{"ANTHROPIC_API_KEY": "test"})
        assert s.temporal_context_enabled is True

    def test_temporal_context_disabled(self, monkeypatch):
        from nous.config import Settings

        monkeypatch.setenv("NOUS_TEMPORAL_CONTEXT_ENABLED", "false")
        s = Settings(**{"ANTHROPIC_API_KEY": "test"})
        assert s.temporal_context_enabled is False


# ---------------------------------------------------------------------------
# TestTemporalContextTier (Phase 2, Task 3)
# ---------------------------------------------------------------------------


class TestTemporalContextTier:
    """Always-on temporal tier injects recent episode titles into context."""

    @pytest.fixture
    def mock_heart(self):
        from unittest.mock import AsyncMock, MagicMock

        heart = MagicMock()
        heart.list_episodes = AsyncMock(return_value=[])
        heart.search_episodes = AsyncMock(return_value=[])
        heart.search_facts = AsyncMock(return_value=[])
        heart.search_procedures = AsyncMock(return_value=[])
        heart.search_censors = AsyncMock(return_value=[])
        heart.list_censors = AsyncMock(return_value=[])
        heart.list_facts_by_category = AsyncMock(return_value=[])
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
        from unittest.mock import MagicMock

        s = MagicMock()
        s.temporal_context_enabled = True
        return s

    @pytest.fixture
    def settings_disabled(self):
        from unittest.mock import MagicMock

        s = MagicMock()
        s.temporal_context_enabled = False
        return s

    @pytest.fixture
    def frame(self):
        from nous.cognitive.schemas import FrameSelection

        return FrameSelection(
            frame_id="task",
            frame_name="Task",
            confidence=1.0,
            match_method="pattern",
        )

    async def test_temporal_tier_includes_recent_titles(
        self, mock_brain, mock_heart, settings, frame
    ):
        """Temporal tier should include recent episode titles in context."""
        from nous.cognitive.context import ContextEngine

        now = datetime.now(UTC)
        mock_heart.list_episodes.return_value = [
            EpisodeSummary(
                id=uuid4(),
                title="Debugging the login flow",
                summary="Fixed auth bug",
                outcome="success",
                started_at=now - timedelta(hours=2),
                tags=["debug"],
            ),
            EpisodeSummary(
                id=uuid4(),
                title="Database migration planning",
                summary="Planned schema changes",
                outcome="success",
                started_at=now - timedelta(hours=5),
                tags=["architecture"],
            ),
        ]

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build("test-agent", "session-1", "hello", frame)

        # Check that the temporal tier section exists
        labels = [s.label for s in result.sections]
        assert "Recent Conversations" in labels

        # Check both titles appear in the system prompt
        assert "Debugging the login flow" in result.system_prompt
        assert "Database migration planning" in result.system_prompt

    async def test_temporal_tier_disabled_by_config(
        self, mock_brain, mock_heart, settings_disabled, frame
    ):
        """When temporal_context_enabled=False, no temporal tier should appear."""
        from nous.cognitive.context import ContextEngine

        now = datetime.now(UTC)
        mock_heart.list_episodes.return_value = [
            EpisodeSummary(
                id=uuid4(),
                title="Should not appear",
                summary="Hidden",
                outcome="success",
                started_at=now - timedelta(hours=1),
                tags=["test"],
            ),
        ]

        engine = ContextEngine(mock_brain, mock_heart, settings_disabled)
        result = await engine.build("test-agent", "session-1", "hello", frame)

        labels = [s.label for s in result.sections]
        assert "Recent Conversations" not in labels

    async def test_temporal_tier_empty_when_no_recent(
        self, mock_brain, mock_heart, settings, frame
    ):
        """When list_episodes returns empty, no temporal section should appear."""
        from nous.cognitive.context import ContextEngine

        mock_heart.list_episodes.return_value = []

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build("test-agent", "session-1", "hello", frame)

        labels = [s.label for s in result.sections]
        assert "Recent Conversations" not in labels

    async def test_temporal_tier_deduplicates_with_semantic(
        self, mock_brain, mock_heart, settings, frame
    ):
        """Episode in temporal tier should be excluded from semantic episode results."""
        from nous.cognitive.context import ContextEngine

        now = datetime.now(UTC)
        shared_id = uuid4()

        # Same episode appears in both temporal and semantic results
        temporal_ep = EpisodeSummary(
            id=shared_id,
            title="Shared episode",
            summary="This appears in both tiers",
            outcome="success",
            started_at=now - timedelta(hours=1),
            tags=["test"],
        )
        semantic_ep = EpisodeSummary(
            id=shared_id,
            title="Shared episode",
            summary="This appears in both tiers",
            outcome="success",
            started_at=now - timedelta(hours=1),
            tags=["test"],
            score=0.9,
        )
        unique_semantic_ep = EpisodeSummary(
            id=uuid4(),
            title="Unique semantic episode",
            summary="Only in semantic search",
            outcome="success",
            started_at=now - timedelta(hours=10),
            tags=["other"],
            score=0.8,
        )

        mock_heart.list_episodes.return_value = [temporal_ep]
        mock_heart.search_episodes.return_value = [semantic_ep, unique_semantic_ep]

        engine = ContextEngine(mock_brain, mock_heart, settings)
        result = await engine.build("test-agent", "session-1", "hello", frame)

        # The shared episode title should appear exactly once (in temporal tier)
        # Count occurrences of "Shared episode" in the prompt
        count = result.system_prompt.count("Shared episode")
        assert count == 1, f"Expected 'Shared episode' once, found {count} times"

        # The unique semantic episode should still appear
        assert "Only in semantic search" in result.system_prompt
