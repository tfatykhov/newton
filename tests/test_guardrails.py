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


# =============================================================================
# Legacy JSONB Backward Compatibility Tests
# =============================================================================


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


# =============================================================================
# CEL Expression Tests
# =============================================================================


async def test_cel_string_condition(session, engine):
    """CEL expression as raw string."""
    # Insert a CEL guardrail with string condition
    guardrail = Guardrail(
        agent_id="test-cel-agent",
        name="cel-string-test",
        condition="decision.stakes == 'high' && decision.confidence < 0.5",
        severity="block",
    )
    session.add(guardrail)
    await session.flush()

    # Should block: high stakes + low confidence
    result = await engine.check(
        session,
        agent_id="test-cel-agent",
        description="Test",
        stakes="high",
        confidence=0.3,
    )
    assert not result.allowed
    assert "cel-string-test" in result.blocked_by


async def test_cel_dict_condition(session, engine):
    """CEL expression in {"cel": "..."} dict format."""
    guardrail = Guardrail(
        agent_id="test-cel-dict-agent",
        name="cel-dict-test",
        condition={"cel": "decision.reason_count < 2"},
        severity="block",
    )
    session.add(guardrail)
    await session.flush()

    # Should block: only 1 reason
    result = await engine.check(
        session,
        agent_id="test-cel-dict-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        reasons=[{"type": "analysis", "text": "one reason"}],
    )
    assert not result.allowed
    assert "cel-dict-test" in result.blocked_by

    # Should allow: 2+ reasons
    result_ok = await engine.check(
        session,
        agent_id="test-cel-dict-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        reasons=[
            {"type": "analysis", "text": "reason 1"},
            {"type": "empirical", "text": "reason 2"},
        ],
    )
    assert result_ok.allowed


async def test_cel_context_access(session, engine):
    """CEL can access decision.context.custom_field."""
    guardrail = Guardrail(
        agent_id="test-cel-context-agent",
        name="require-architecture-review",
        condition={"cel": "decision.category == 'architecture' && !decision.context.architecture_review"},
        severity="block",
    )
    session.add(guardrail)
    await session.flush()

    # Should block: architecture category without review
    result = await engine.check(
        session,
        agent_id="test-cel-context-agent",
        description="Architecture decision",
        stakes="high",
        confidence=0.9,
        category="architecture",
        context={},
    )
    assert not result.allowed
    assert "require-architecture-review" in result.blocked_by

    # Should allow: architecture with review flag
    result_ok = await engine.check(
        session,
        agent_id="test-cel-context-agent",
        description="Architecture decision",
        stakes="high",
        confidence=0.9,
        category="architecture",
        context={"architecture_review": True},
    )
    assert result_ok.allowed


async def test_cel_tag_check(session, engine):
    """CEL can check tags with size() function."""
    guardrail = Guardrail(
        agent_id="test-cel-tags-agent",
        name="require-tags",
        condition={"cel": "size(decision.tags) == 0"},
        severity="warn",
    )
    session.add(guardrail)
    await session.flush()

    # Should warn: no tags
    result = await engine.check(
        session,
        agent_id="test-cel-tags-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        tags=[],
    )
    assert result.allowed  # warn doesn't block
    assert "require-tags" in result.warnings

    # Should not warn: has tags
    result_ok = await engine.check(
        session,
        agent_id="test-cel-tags-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        tags=["tag1", "tag2"],
    )
    assert "require-tags" not in result_ok.warnings


async def test_cel_pattern_check(session, engine):
    """CEL can check decision.has_pattern boolean."""
    guardrail = Guardrail(
        agent_id="test-cel-pattern-agent",
        name="require-pattern",
        condition={"cel": "!decision.has_pattern"},
        severity="warn",
    )
    session.add(guardrail)
    await session.flush()

    # Should warn: no pattern
    result = await engine.check(
        session,
        agent_id="test-cel-pattern-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        pattern=None,
    )
    assert result.allowed
    assert "require-pattern" in result.warnings

    # Should not warn: has pattern
    result_ok = await engine.check(
        session,
        agent_id="test-cel-pattern-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        pattern="some pattern",
    )
    assert "require-pattern" not in result_ok.warnings


async def test_cel_complex_expression(session, engine):
    """CEL complex expression with multiple conditions."""
    guardrail = Guardrail(
        agent_id="test-cel-complex-agent",
        name="critical-high-confidence-reviewed",
        condition={
            "cel": "(decision.stakes == 'critical' || decision.stakes == 'high') && "
            "decision.confidence > 0.8 && "
            "!decision.context.reviewed"
        },
        severity="block",
    )
    session.add(guardrail)
    await session.flush()

    # Should block: critical + high confidence + not reviewed
    result = await engine.check(
        session,
        agent_id="test-cel-complex-agent",
        description="Test",
        stakes="critical",
        confidence=0.9,
        context={},
    )
    assert not result.allowed

    # Should allow: critical + high confidence + reviewed
    result_ok = await engine.check(
        session,
        agent_id="test-cel-complex-agent",
        description="Test",
        stakes="critical",
        confidence=0.9,
        context={"reviewed": True},
    )
    assert result_ok.allowed


async def test_cel_invalid_syntax_fail_closed(session, engine):
    """Invalid CEL syntax should fail-closed for block severity."""
    guardrail = Guardrail(
        agent_id="test-cel-invalid-agent",
        name="invalid-syntax",
        condition={"cel": "this is not valid CEL syntax!!!"},
        severity="block",
    )
    session.add(guardrail)
    await session.flush()

    # Should block (fail-closed) because CEL is invalid
    result = await engine.check(
        session,
        agent_id="test-cel-invalid-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
    )
    # Fail-closed: invalid expression should trigger block
    assert not result.allowed
    assert "invalid-syntax" in result.blocked_by


async def test_cel_invalid_syntax_fail_open_warn(session, engine):
    """Invalid CEL syntax should fail-open for warn severity."""
    guardrail = Guardrail(
        agent_id="test-cel-invalid-warn-agent",
        name="invalid-syntax-warn",
        condition={"cel": "this is not valid CEL syntax!!!"},
        severity="warn",
    )
    session.add(guardrail)
    await session.flush()

    # Should allow (fail-open for warn) because CEL is invalid
    result = await engine.check(
        session,
        agent_id="test-cel-invalid-warn-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
    )
    # Fail-open for warn: invalid expression should not trigger
    assert result.allowed
    assert "invalid-syntax-warn" not in result.warnings


async def test_priority_ordering(session, engine):
    """Guardrails evaluate in priority order (low number = high priority)."""
    # Insert guardrails with different priorities
    guardrails = [
        Guardrail(
            agent_id="test-priority-agent",
            name="priority-50",
            condition={"cel": "decision.stakes == 'high'"},
            severity="block",
            priority=50,
        ),
        Guardrail(
            agent_id="test-priority-agent",
            name="priority-100",
            condition={"cel": "decision.confidence < 0.5"},
            severity="block",
            priority=100,
        ),
        Guardrail(
            agent_id="test-priority-agent",
            name="priority-10",
            condition={"cel": "decision.reason_count < 1"},
            severity="block",
            priority=10,
        ),
    ]
    for g in guardrails:
        session.add(g)
    await session.flush()

    # All three conditions match
    result = await engine.check(
        session,
        agent_id="test-priority-agent",
        description="Test",
        stakes="high",
        confidence=0.3,
        reasons=[],
    )

    # All should trigger
    assert not result.allowed
    assert len(result.blocked_by) == 3

    # Order in blocked_by should match priority (ascending)
    # priority-10, priority-50, priority-100
    assert result.blocked_by[0] == "priority-10"
    assert result.blocked_by[1] == "priority-50"
    assert result.blocked_by[2] == "priority-100"


async def test_validate_expression(engine):
    """validate_expression() detects valid and invalid CEL."""
    # Valid expression
    valid, error = engine.validate_expression("decision.stakes == 'high'")
    assert valid
    assert error is None

    # Invalid expression
    valid, error = engine.validate_expression("this is not CEL!!!")
    assert not valid
    assert error is not None
    assert len(error) > 0


async def test_cel_reasons_access(session, engine):
    """CEL can access individual reasons with type and text."""
    guardrail = Guardrail(
        agent_id="test-cel-reasons-agent",
        name="require-analysis-reason",
        condition={"cel": "size([r for r in decision.reasons if r.type == 'analysis']) == 0"},
        severity="warn",
    )
    session.add(guardrail)
    await session.flush()

    # Should warn: no analysis reason
    result = await engine.check(
        session,
        agent_id="test-cel-reasons-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        reasons=[
            {"type": "intuition", "text": "gut feeling"},
        ],
    )
    assert result.allowed
    assert "require-analysis-reason" in result.warnings

    # Should not warn: has analysis reason
    result_ok = await engine.check(
        session,
        agent_id="test-cel-reasons-agent",
        description="Test",
        stakes="low",
        confidence=0.9,
        reasons=[
            {"type": "analysis", "text": "analytical reason"},
        ],
    )
    assert "require-analysis-reason" not in result_ok.warnings


async def test_cel_evaluation_timeout(engine):
    """Timeout mechanism works for CEL evaluation."""
    # Create a very complex nested expression that might take time
    # Note: This test verifies the timeout mechanism exists and handles gracefully
    # In practice, CEL evaluations are fast, so actual timeout is rare
    complex_expr = " && ".join([f"decision.confidence > {i / 100.0}" for i in range(100)])

    valid, error = engine.validate_expression(complex_expr)
    # Should still validate successfully (syntax is valid)
    assert valid

    # Even complex expressions should evaluate quickly with ThreadPoolExecutor
    # The timeout protection ensures we don't hang on pathological cases
    activation = {"decision": {"confidence": 0.5, "stakes": "low"}}
    result = engine._evaluate(complex_expr, activation, severity="block")
    # Should return a boolean (either True/False or fail-closed=True for timeout)
    assert isinstance(result, bool)
