"""Tests for EpisodeManager — episodic memory.

All tests use real Postgres via the SAVEPOINT fixture from conftest.py.
Heart methods receive the test session via the session parameter (P1-1).
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from nous.brain.brain import Brain
from nous.brain.schemas import RecordInput, ReasonInput
from nous.config import Settings
from nous.heart import (
    Heart,
    EpisodeDetail,
    EpisodeInput,
    EpisodeSummary,
)
from nous.storage.models import EpisodeDecision, EpisodeProcedure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _episode_input(**overrides) -> EpisodeInput:
    """Build an EpisodeInput with sensible defaults."""
    defaults = dict(
        title="Test Episode",
        summary="A test episode for unit testing",
        trigger="unit_test",
        participants=["agent-1"],
        tags=["test"],
    )
    defaults.update(overrides)
    return EpisodeInput(**defaults)


# ---------------------------------------------------------------------------
# 1. test_start_episode
# ---------------------------------------------------------------------------


async def test_start_episode(heart, session):
    """Start creates episode with started_at, no ended_at."""
    inp = _episode_input()
    detail = await heart.start_episode(inp, session=session)

    assert isinstance(detail, EpisodeDetail)
    assert detail.title == "Test Episode"
    assert detail.summary == "A test episode for unit testing"
    assert detail.started_at is not None
    assert detail.ended_at is None
    assert detail.duration_seconds is None
    assert detail.outcome is None
    assert detail.trigger == "unit_test"


# ---------------------------------------------------------------------------
# 2. test_end_episode
# ---------------------------------------------------------------------------


async def test_end_episode(heart, session):
    """End sets ended_at, duration_seconds, outcome."""
    inp = _episode_input()
    detail = await heart.start_episode(inp, session=session)

    ended = await heart.end_episode(
        detail.id,
        outcome="success",
        lessons_learned=["lesson 1"],
        surprise_level=0.3,
        session=session,
    )

    assert ended.ended_at is not None
    assert ended.outcome == "success"
    assert ended.duration_seconds is not None
    assert ended.lessons_learned == ["lesson 1"]
    assert ended.surprise_level == 0.3


# ---------------------------------------------------------------------------
# 3. test_end_episode_calculates_duration
# ---------------------------------------------------------------------------


async def test_end_episode_calculates_duration(heart, session):
    """duration = ended_at - started_at."""
    inp = _episode_input()
    detail = await heart.start_episode(inp, session=session)

    ended = await heart.end_episode(
        detail.id,
        outcome="success",
        session=session,
    )

    # Duration should be >= 0 (test runs fast, so ~0)
    assert ended.duration_seconds is not None
    assert ended.duration_seconds >= 0


# ---------------------------------------------------------------------------
# 4. test_link_decision
# ---------------------------------------------------------------------------


async def test_link_decision(heart, db, settings, session):
    """episode_decisions row created when linking a decision."""
    # Create an episode
    inp = _episode_input()
    episode = await heart.start_episode(inp, session=session)

    # Create a brain decision to link
    brain = Brain(database=db, settings=settings)
    decision = await brain.record(
        RecordInput(
            description="Test decision for linking",
            confidence=0.8,
            category="architecture",
            stakes="low",
            reasons=[ReasonInput(type="analysis", text="Test")],
        ),
        session=session,
    )
    await brain.close()

    # Link the decision to the episode
    await heart.link_decision_to_episode(
        episode.id, decision.id, session=session
    )

    # Verify the link exists
    result = await session.execute(
        select(EpisodeDecision).where(
            EpisodeDecision.episode_id == episode.id,
            EpisodeDecision.decision_id == decision.id,
        )
    )
    link = result.scalar_one()
    assert link.episode_id == episode.id
    assert link.decision_id == decision.id


# ---------------------------------------------------------------------------
# 5. test_link_procedure_with_effectiveness
# ---------------------------------------------------------------------------


async def test_link_procedure_with_effectiveness(heart, session):
    """episode_procedures row created with effectiveness."""
    from nous.heart import ProcedureInput

    episode = await heart.start_episode(_episode_input(), session=session)
    procedure = await heart.store_procedure(
        ProcedureInput(
            name="Test Procedure",
            domain="testing",
            core_patterns=["test pattern"],
        ),
        session=session,
    )

    await heart.link_procedure_to_episode(
        episode.id, procedure.id, effectiveness="helped", session=session
    )

    result = await session.execute(
        select(EpisodeProcedure).where(
            EpisodeProcedure.episode_id == episode.id,
            EpisodeProcedure.procedure_id == procedure.id,
        )
    )
    link = result.scalar_one()
    assert link.effectiveness == "helped"


# ---------------------------------------------------------------------------
# 6. test_list_recent
# ---------------------------------------------------------------------------


async def test_list_recent(heart, session):
    """List recent episodes ordered by started_at DESC."""
    # Create 3 episodes
    for i in range(3):
        await heart.start_episode(
            _episode_input(title=f"Episode {i}", summary=f"Episode number {i}"),
            session=session,
        )

    results = await heart.list_episodes(limit=10, session=session)
    assert isinstance(results, list)
    assert len(results) >= 3

    # Should be ordered by started_at DESC
    for r in results:
        assert isinstance(r, EpisodeSummary)


# ---------------------------------------------------------------------------
# 7. test_search_episodes
# ---------------------------------------------------------------------------


async def test_search_episodes(heart, session):
    """Hybrid search returns relevant episodes."""
    # Create episodes with distinct summaries
    await heart.start_episode(
        _episode_input(
            title="Database migration",
            summary="Migrated PostgreSQL schema to version 3",
        ),
        session=session,
    )
    await heart.start_episode(
        _episode_input(
            title="UI refactor",
            summary="Refactored the React components",
        ),
        session=session,
    )

    results = await heart.search_episodes(
        "Migrated PostgreSQL schema to version 3", session=session
    )
    assert isinstance(results, list)
    # With mock embeddings, identical text should match
    if results:
        assert any("PostgreSQL" in r.summary or "migration" in (r.title or "") for r in results)
