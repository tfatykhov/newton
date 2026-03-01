"""Tests for DeliberationEngine — deliberation lifecycle wrapper.

All tests use real Postgres via the SAVEPOINT fixture from conftest.py.
DeliberationEngine wraps Brain.record(), Brain.think(), Brain.update()
into a clean start -> think -> finalize flow.
"""

import uuid

import pytest_asyncio

from nous.brain.brain import Brain
from nous.cognitive.deliberation import DeliberationEngine
from nous.cognitive.schemas import FrameSelection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def brain(db, settings):
    """Brain without embeddings for deliberation tests."""
    b = Brain(database=db, settings=settings)
    yield b
    await b.close()


@pytest_asyncio.fixture
async def delib(brain):
    """DeliberationEngine wired to Brain."""
    return DeliberationEngine(brain)


def _frame(frame_id: str = "decision", **overrides) -> FrameSelection:
    """Build a FrameSelection with defaults."""
    defaults = dict(
        frame_id=frame_id,
        frame_name="Decision Making",
        confidence=0.9,
        match_method="pattern",
        default_category="architecture",
        default_stakes="high",
    )
    defaults.update(overrides)
    return FrameSelection(**defaults)


# ---------------------------------------------------------------------------
# 1. test_start_creates_decision
# ---------------------------------------------------------------------------


async def test_start_creates_decision(delib, brain, session):
    """start() records a decision with 'Plan: ...' prefix."""
    frame = _frame()
    decision_id = await delib.start(
        "nous-default",
        "evaluate database options",
        frame,
        session=session,
    )

    assert decision_id is not None
    assert isinstance(decision_id, str)

    # Verify the decision was created in Brain
    detail = await brain.get(uuid.UUID(decision_id), session=session)
    assert detail is not None
    assert detail.description.startswith("Plan:")
    assert "evaluate database options" in detail.description


# ---------------------------------------------------------------------------
# 2. test_start_uses_frame_defaults
# ---------------------------------------------------------------------------


async def test_start_uses_frame_defaults(delib, brain, session):
    """start() uses frame's default_category and default_stakes."""
    frame = _frame(default_category="tooling", default_stakes="medium")
    decision_id = await delib.start("nous-default", "set up CI pipeline", frame, session=session)

    detail = await brain.get(uuid.UUID(decision_id), session=session)
    assert detail.category == "tooling"
    assert detail.stakes == "medium"
    # Initial confidence is 0.5
    assert detail.confidence == 0.5


# ---------------------------------------------------------------------------
# 3. test_think_delegates_to_brain
# ---------------------------------------------------------------------------


async def test_think_delegates_to_brain(delib, brain, session):
    """think() records a thought via Brain.think()."""
    frame = _frame()
    decision_id = await delib.start("nous-default", "think test", frame, session=session)

    await delib.think(decision_id, "Redis is fast but single-threaded", "nous-default", session=session)

    # Verify thought exists on the decision
    await session.flush()
    session.expire_all()
    detail = await brain.get(uuid.UUID(decision_id), session=session)
    assert len(detail.thoughts) == 1
    assert "Redis" in detail.thoughts[0].text


# ---------------------------------------------------------------------------
# 4. test_finalize_updates_decision
# ---------------------------------------------------------------------------


async def test_finalize_updates_decision(delib, brain, session):
    """finalize() updates decision description and confidence via Brain.update()."""
    frame = _frame()
    decision_id = await delib.start("nous-default", "finalize test", frame, session=session)

    await delib.finalize(
        decision_id,
        description="Final: Use PostgreSQL for storage",
        confidence=0.8,
        context="After evaluating all options",
        session=session,
    )

    detail = await brain.get(uuid.UUID(decision_id), session=session)
    assert "Final:" in detail.description or "PostgreSQL" in detail.description
    # Confidence updated from initial 0.5 to 0.8
    assert detail.confidence == 0.8


# ---------------------------------------------------------------------------
# 5. test_should_deliberate_decision
# ---------------------------------------------------------------------------


async def test_should_deliberate_decision(delib):
    """Decision frame triggers deliberation."""
    frame = _frame("decision")
    result = await delib.should_deliberate(frame)
    assert result is True


# ---------------------------------------------------------------------------
# 6. test_should_deliberate_task
# ---------------------------------------------------------------------------


async def test_should_deliberate_task(delib):
    """Task frame does NOT trigger deliberation (009.5)."""
    frame = _frame("task", frame_name="Task Execution")
    result = await delib.should_deliberate(frame)
    assert result is False


# ---------------------------------------------------------------------------
# 7. test_should_deliberate_conversation
# ---------------------------------------------------------------------------


async def test_should_deliberate_conversation(delib):
    """Conversation frame does NOT trigger deliberation."""
    frame = _frame("conversation", frame_name="Conversation")
    result = await delib.should_deliberate(frame)
    assert result is False


# ---------------------------------------------------------------------------
# 8. test_get_recent_decisions_returns_recent (009.5 Task 4)
# ---------------------------------------------------------------------------


async def test_get_recent_decisions_returns_recent(brain, session):
    """009.5: get_recent_decisions() returns decisions created with a session_id."""
    from datetime import UTC, datetime, timedelta

    from nous.brain.schemas import ReasonInput, RecordInput

    record_input = RecordInput(
        description="Pick a database engine",
        confidence=0.7,
        category="architecture",
        stakes="high",
        reasons=[ReasonInput(type="analysis", text="Evaluating options")],
        session_id="session-alpha",
    )
    detail = await brain.record(record_input, session=session)

    cutoff = datetime.now(UTC) - timedelta(minutes=5)
    results = await brain.get_recent_decisions(
        "nous-default", since=cutoff, session_id="session-alpha", session=session,
    )
    assert len(results) >= 1
    assert any(r.id == detail.id for r in results)


# ---------------------------------------------------------------------------
# 9. test_get_recent_decisions_filters_by_session (009.5 Task 4)
# ---------------------------------------------------------------------------


async def test_get_recent_decisions_filters_by_session(brain, session):
    """009.5: get_recent_decisions() only returns decisions from the queried session."""
    from datetime import UTC, datetime, timedelta

    from nous.brain.schemas import ReasonInput, RecordInput

    input_a = RecordInput(
        description="Decision in session A",
        confidence=0.6,
        category="process",
        stakes="low",
        reasons=[ReasonInput(type="analysis", text="Session A work")],
        session_id="session-A",
    )
    input_b = RecordInput(
        description="Decision in session B",
        confidence=0.6,
        category="process",
        stakes="low",
        reasons=[ReasonInput(type="analysis", text="Session B work")],
        session_id="session-B",
    )
    await brain.record(input_a, session=session)
    detail_b = await brain.record(input_b, session=session)

    cutoff = datetime.now(UTC) - timedelta(minutes=5)
    results = await brain.get_recent_decisions(
        "nous-default", since=cutoff, session_id="session-B", session=session,
    )
    assert len(results) == 1
    assert results[0].id == detail_b.id
