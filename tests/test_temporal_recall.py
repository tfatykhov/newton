"""Tests for temporal recall — spec 008.6 Phase 1.

Covers the list_recent() fix (active filter → ended_at filter)
and the new `hours` parameter for time-windowed episode listing.

5 tests in TestListRecentFix.
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
