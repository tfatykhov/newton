"""Unit tests for guardrail evaluation engine."""

import pytest
from sqlalchemy import select

from nous.brain.guardrails import GuardrailEngine
from nous.storage.models import Guardrail

# Must match conftest.GUARDRAIL_TEST_AGENT
GUARDRAIL_TEST_AGENT = "test-guardrail-agent"


@pytest.fixture
def engine() -> GuardrailEngine:
    return GuardrailEngine()


async def test_stakes_match(session, seed_guardrails, engine):
    """Condition {"stakes": "high"} matches high stakes."""
    result = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="Test decision",
        stakes="high",
        confidence=0.3,  # Low confidence + high stakes -> triggers no-high-stakes-low-confidence
        reasons=[{"type": "analysis", "text": "reason"}],
        quality_score=0.8,
    )
    assert not result.allowed
    assert "no-high-stakes-low-confidence" in result.blocked_by


async def test_confidence_lt(session, seed_guardrails, engine):
    """Condition {"confidence_lt": 0.5} matches 0.3 but not 0.8."""
    # High stakes + low confidence -> blocked
    result_low = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="Low confidence decision",
        stakes="high",
        confidence=0.3,
        reasons=[{"type": "analysis", "text": "reason"}],
        quality_score=0.8,
    )
    assert "no-high-stakes-low-confidence" in result_low.blocked_by

    # High stakes + high confidence -> not blocked by this guardrail
    result_high = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="High confidence decision",
        stakes="high",
        confidence=0.8,
        reasons=[{"type": "analysis", "text": "reason"}],
        quality_score=0.8,
    )
    assert "no-high-stakes-low-confidence" not in result_high.blocked_by


async def test_reason_count_lt(session, seed_guardrails, engine):
    """Condition {"reason_count_lt": 1} matches empty reasons."""
    result = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="No reasons decision",
        stakes="low",
        confidence=0.9,
        reasons=[],
        quality_score=0.8,
    )
    assert "require-reasons" in result.blocked_by

    # With one reason -> not blocked
    result_ok = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="Has reasons",
        stakes="low",
        confidence=0.9,
        reasons=[{"type": "analysis", "text": "reason"}],
        quality_score=0.8,
    )
    assert "require-reasons" not in result_ok.blocked_by


async def test_quality_lt(session, seed_guardrails, engine):
    """Condition {"quality_lt": 0.5} matches low quality."""
    result = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="Low quality decision",
        stakes="low",
        confidence=0.9,
        reasons=[{"type": "analysis", "text": "reason"}],
        quality_score=0.3,
    )
    assert "low-quality-recording" in result.blocked_by

    # High quality -> not blocked
    result_ok = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="High quality",
        stakes="low",
        confidence=0.9,
        reasons=[{"type": "analysis", "text": "reason"}],
        quality_score=0.8,
    )
    assert "low-quality-recording" not in result_ok.blocked_by


async def test_multiple_conditions(session, seed_guardrails, engine):
    """Both conditions must match (AND logic) for guardrail to trigger.

    no-high-stakes-low-confidence requires BOTH stakes=high AND confidence_lt=0.5.
    High stakes with high confidence should NOT trigger it.
    """
    result = await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="High stakes, high confidence",
        stakes="high",
        confidence=0.9,
        reasons=[{"type": "analysis", "text": "reason"}],
        quality_score=0.8,
    )
    # Stakes matches "high" but confidence >= 0.5 does NOT match confidence_lt
    assert "no-high-stakes-low-confidence" not in result.blocked_by


async def test_warn_vs_block(session, engine):
    """Warn severity allows action, block severity denies."""
    # Insert a warn guardrail
    warn_guardrail = Guardrail(
        agent_id="test-warn-agent",
        name="warn-test",
        condition={"stakes": "high"},
        severity="warn",
    )
    session.add(warn_guardrail)
    await session.flush()

    result = await engine.check(
        session,
        agent_id="test-warn-agent",
        description="Should warn, not block",
        stakes="high",
        confidence=0.9,
        reasons=[{"type": "analysis", "text": "reason"}],
    )
    assert result.allowed, "Warn severity should allow the action"
    assert "warn-test" in result.warnings


async def test_inactive_guardrail_skipped(session, engine):
    """Disabled guardrails don't trigger."""
    # Insert an inactive guardrail
    inactive = Guardrail(
        agent_id="test-inactive-agent",
        name="inactive-rule",
        condition={"stakes": "low"},
        severity="block",
        active=False,
    )
    session.add(inactive)
    await session.flush()

    result = await engine.check(
        session,
        agent_id="test-inactive-agent",
        description="Should not be blocked",
        stakes="low",
        confidence=0.9,
        reasons=[{"type": "analysis", "text": "reason"}],
    )
    assert result.allowed
    assert len(result.blocked_by) == 0


async def test_activation_count_incremented(session, seed_guardrails, engine):
    """Counter goes up on trigger."""
    # Trigger the require-reasons guardrail
    await engine.check(
        session,
        agent_id=GUARDRAIL_TEST_AGENT,
        description="No reasons",
        stakes="low",
        confidence=0.9,
        reasons=[],
        quality_score=0.8,
    )

    # Check activation_count was incremented
    result = await session.execute(
        select(Guardrail).where(
            Guardrail.agent_id == GUARDRAIL_TEST_AGENT,
            Guardrail.name == "require-reasons",
        )
    )
    guardrail = result.scalar_one()
    assert guardrail.activation_count >= 1
    assert guardrail.last_activated is not None
