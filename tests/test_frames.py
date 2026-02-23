"""Tests for FrameEngine — cognitive frame selection via pattern matching.

All tests use real Postgres via the SAVEPOINT fixture from conftest.py.
Frames are pre-seeded in seed.sql (6 default frames for nous-default agent).
"""

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from nous.cognitive.frames import FrameEngine
from nous.cognitive.schemas import FrameSelection
from nous.config import Settings
from nous.storage.models import Frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def frame_engine(db, settings):
    """FrameEngine wired to test database."""
    return FrameEngine(db, settings)


# ---------------------------------------------------------------------------
# 1. test_frame_select_decision
# ---------------------------------------------------------------------------


async def test_frame_select_decision(frame_engine, session):
    """Input with 'should' keyword selects decision frame."""
    result = await frame_engine.select("nous-default", "should we use Redis?", session=session)
    assert isinstance(result, FrameSelection)
    assert result.frame_id == "decision"
    assert result.match_method == "pattern"
    assert result.confidence > 0


# ---------------------------------------------------------------------------
# 2. test_frame_select_task
# ---------------------------------------------------------------------------


async def test_frame_select_task(frame_engine, session):
    """Input with 'build' keyword selects task frame."""
    result = await frame_engine.select("nous-default", "build a REST API", session=session)
    assert result.frame_id == "task"
    assert result.match_method == "pattern"


# ---------------------------------------------------------------------------
# 3. test_frame_select_debug
# ---------------------------------------------------------------------------


async def test_frame_select_debug(frame_engine, session):
    """Input with 'error' keyword selects debug frame."""
    result = await frame_engine.select("nous-default", "error in deployment", session=session)
    assert result.frame_id == "debug"
    assert result.match_method == "pattern"


# ---------------------------------------------------------------------------
# 4. test_frame_select_conversation — multi-word pattern (P2-1)
# ---------------------------------------------------------------------------


async def test_frame_select_conversation(frame_engine, session):
    """Input 'hello how are you' matches conversation with 2 patterns.

    P2-1 fix: multi-word activation patterns checked as substrings.
    'hello' (single-word) + 'how are you' (multi-word) = 2 matches for conversation.
    'how' also matches question (1 match), but conversation wins on count.
    """
    result = await frame_engine.select("nous-default", "hello how are you", session=session)
    assert result.frame_id == "conversation"
    assert result.match_method == "pattern"


# ---------------------------------------------------------------------------
# 5. test_frame_select_no_match
# ---------------------------------------------------------------------------


async def test_frame_select_no_match(frame_engine, session):
    """Input with no matching keywords falls back to default conversation."""
    result = await frame_engine.select("nous-default", "xyzzy foobar", session=session)
    assert result.frame_id == "conversation"
    assert result.match_method == "default"


# ---------------------------------------------------------------------------
# 6. test_frame_select_tiebreak
# ---------------------------------------------------------------------------


async def test_frame_select_tiebreak(frame_engine, session):
    """'should we fix this bug' matches decision ('should') and debug ('bug').

    Decision has higher priority than debug, so decision wins.
    """
    result = await frame_engine.select(
        "nous-default", "should we fix this bug", session=session
    )
    # Both 'should' (decision) and 'bug' (debug) match — decision wins tiebreak
    assert result.frame_id == "decision"


# ---------------------------------------------------------------------------
# 7. test_frame_list
# ---------------------------------------------------------------------------


async def test_frame_list(frame_engine, session):
    """Lists all 6 seed frames for the default agent."""
    frames = await frame_engine.list_frames("nous-default", session=session)
    assert len(frames) == 6
    frame_ids = {f.frame_id for f in frames}
    assert frame_ids == {"task", "question", "decision", "creative", "conversation", "debug"}


# ---------------------------------------------------------------------------
# 8. test_frame_get
# ---------------------------------------------------------------------------


async def test_frame_get(frame_engine, session):
    """Fetch a specific frame by ID."""
    result = await frame_engine.get("task", "nous-default", session=session)
    assert isinstance(result, FrameSelection)
    assert result.frame_id == "task"
    assert result.frame_name == "Task Execution"
    assert result.default_category == "tooling"


# ---------------------------------------------------------------------------
# 9. test_frame_usage_count_increments
# ---------------------------------------------------------------------------


async def test_frame_usage_count_increments(frame_engine, session):
    """Selecting a frame bumps its usage_count."""
    # Get initial count
    result = await session.execute(
        select(Frame).where(Frame.id == "task", Frame.agent_id == "nous-default")
    )
    frame = result.scalar_one()
    initial_count = frame.usage_count or 0

    # Select the frame
    await frame_engine.select("nous-default", "build something", session=session)

    # Verify count incremented
    await session.flush()
    session.expire_all()
    result2 = await session.execute(
        select(Frame).where(Frame.id == "task", Frame.agent_id == "nous-default")
    )
    frame2 = result2.scalar_one()
    assert (frame2.usage_count or 0) == initial_count + 1
