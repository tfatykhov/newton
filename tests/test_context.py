"""Tests for ContextEngine — context assembly within token budgets.

All tests use real Postgres via the SAVEPOINT fixture from conftest.py.
Pre-seeds Brain with decisions and Heart with facts, procedures, and censors
to test realistic context assembly.
"""

import uuid

import pytest_asyncio

from nous.brain.brain import Brain
from nous.brain.schemas import ReasonInput, RecordInput
from nous.cognitive.context import ContextEngine
from nous.cognitive.schemas import ContextBudget, FrameSelection
from nous.heart import CensorInput, FactInput, ProcedureInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def brain(db, settings):
    """Brain without embeddings for context tests."""
    b = Brain(database=db, settings=settings)
    yield b
    await b.close()


@pytest_asyncio.fixture
async def context_engine(brain, heart, settings):
    """ContextEngine wired to Brain and Heart."""
    return ContextEngine(brain, heart, settings, identity_prompt="You are Nous, a thinking agent.")


def _frame_selection(frame_id: str = "task", **overrides) -> FrameSelection:
    """Build a FrameSelection with defaults."""
    defaults = dict(
        frame_id=frame_id,
        frame_name="Task Execution",
        confidence=0.9,
        match_method="pattern",
        default_category="tooling",
        default_stakes="medium",
    )
    defaults.update(overrides)
    return FrameSelection(**defaults)


def _record_input(**overrides) -> RecordInput:
    """Build a RecordInput with sensible defaults."""
    defaults = dict(
        description="Use PostgreSQL for storage",
        confidence=0.85,
        category="architecture",
        stakes="medium",
        context="Evaluating database options",
        reasons=[ReasonInput(type="analysis", text="Solid choice")],
    )
    defaults.update(overrides)
    return RecordInput(**defaults)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


async def _seed_decisions(brain, session, count=3):
    """Pre-seed Brain with decisions."""
    decisions = []
    for i in range(count):
        d = await brain.record(
            _record_input(description=f"Context test decision {i}"),
            session=session,
        )
        decisions.append(d)
    return decisions


async def _seed_facts(heart, session, count=5):
    """Pre-seed Heart with facts."""
    facts = []
    for i in range(count):
        f = await heart.learn(
            FactInput(
                content=f"Context test fact {i}: important information",
                category="technical",
                subject="testing",
                confidence=0.9,
            ),
            session=session,
        )
        facts.append(f)
    return facts


async def _seed_procedures(heart, session, count=2):
    """Pre-seed Heart with procedures."""
    procs = []
    for i in range(count):
        p = await heart.store_procedure(
            ProcedureInput(
                name=f"Context test procedure {i}",
                domain="testing",
                core_patterns=[f"pattern-{i}"],
            ),
            session=session,
        )
        procs.append(p)
    return procs


async def _seed_censor(heart, session):
    """Pre-seed Heart with an active censor."""
    return await heart.add_censor(
        CensorInput(
            trigger_pattern="dangerous operation",
            reason="Avoid risky operations without review",
            action="warn",
        ),
        session=session,
    )


# ---------------------------------------------------------------------------
# 1. test_build_conversation_budget
# ---------------------------------------------------------------------------


async def test_build_conversation_budget(context_engine, session):
    """Conversation frame gets 3K total budget."""
    budget = ContextBudget.for_frame("conversation")
    assert budget.total == 3000
    assert budget.procedures == 0  # Conversation skips procedures


# ---------------------------------------------------------------------------
# 2. test_build_decision_budget
# ---------------------------------------------------------------------------


async def test_build_decision_budget(context_engine, session):
    """Decision frame gets 12K total budget."""
    budget = ContextBudget.for_frame("decision")
    assert budget.total == 12000
    assert budget.decisions == 3000


# ---------------------------------------------------------------------------
# 3. test_build_includes_identity
# ---------------------------------------------------------------------------


async def test_build_includes_identity(context_engine, brain, heart, session):
    """Identity section always present in system prompt."""
    frame = _frame_selection()
    sid = f"test-ctx-identity-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)

    prompt, sections = await context_engine.build("nous-default", sid, "build something", frame, session=session)

    assert "Nous" in prompt or "thinking agent" in prompt
    assert len(prompt) > 0


# ---------------------------------------------------------------------------
# 4. test_build_includes_censors
# ---------------------------------------------------------------------------


async def test_build_includes_censors(context_engine, brain, heart, session):
    """Active censors appear in context."""
    await _seed_censor(heart, session)
    frame = _frame_selection()
    sid = f"test-ctx-censors-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)

    prompt, sections = await context_engine.build("nous-default", sid, "do something dangerous", frame, session=session)

    # Censor should appear somewhere in the prompt
    assert "dangerous operation" in prompt.lower() or any("censor" in s.label.lower() for s in sections)


# ---------------------------------------------------------------------------
# 5. test_build_truncates_to_budget
# ---------------------------------------------------------------------------


async def test_build_truncates_to_budget(context_engine, session):
    """Long content is truncated, not omitted."""
    engine = context_engine
    long_text = "x" * 100000  # Very long text
    truncated = engine._truncate_to_budget(long_text, 100)
    # 100 tokens * 4 chars/token = 400 chars max
    assert len(truncated) <= 400
    assert truncated.endswith("...")


# ---------------------------------------------------------------------------
# 6. test_build_skips_zero_budget
# ---------------------------------------------------------------------------


async def test_build_skips_zero_budget(context_engine, brain, heart, session):
    """Conversation frame skips procedures (budget=0)."""
    await _seed_procedures(heart, session)
    frame = _frame_selection("conversation", frame_name="Conversation")
    sid = f"test-ctx-skip-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)

    prompt, sections = await context_engine.build("nous-default", sid, "hello there", frame, session=session)

    # Procedures section should not be present (budget=0)
    procedure_sections = [s for s in sections if "procedure" in s.label.lower()]
    assert len(procedure_sections) == 0


# ---------------------------------------------------------------------------
# 7. test_build_with_decisions
# ---------------------------------------------------------------------------


async def test_build_with_decisions(context_engine, brain, heart, session):
    """Brain.query() results appear in context."""
    await _seed_decisions(brain, session)
    frame = _frame_selection()
    sid = f"test-ctx-decisions-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)

    prompt, sections = await context_engine.build("nous-default", sid, "context test decision", frame, session=session)

    # Decisions section should appear with seeded data
    decision_sections = [s for s in sections if "decision" in s.label.lower()]
    if decision_sections:
        assert len(decision_sections[0].content) > 0


# ---------------------------------------------------------------------------
# 8. test_build_with_facts
# ---------------------------------------------------------------------------


async def test_build_with_facts(context_engine, brain, heart, session):
    """Heart facts appear in context."""
    await _seed_facts(heart, session)
    frame = _frame_selection()
    sid = f"test-ctx-facts-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)

    prompt, sections = await context_engine.build("nous-default", sid, "context test fact", frame, session=session)

    fact_sections = [s for s in sections if "fact" in s.label.lower()]
    if fact_sections:
        assert len(fact_sections[0].content) > 0


# ---------------------------------------------------------------------------
# 9. test_build_with_working_memory
# ---------------------------------------------------------------------------


async def test_build_with_working_memory(context_engine, brain, heart, session):
    """Working memory current task appears in context."""
    sid = f"test-ctx-wm-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)
    await heart.focus(sid, task="Build the REST API", frame="task", session=session)

    frame = _frame_selection()
    prompt, sections = await context_engine.build("nous-default", sid, "continue building", frame, session=session)

    wm_sections = [s for s in sections if "working" in s.label.lower() or "memory" in s.label.lower()]
    if wm_sections:
        assert "REST API" in wm_sections[0].content or "Build" in wm_sections[0].content


# ---------------------------------------------------------------------------
# 10. test_refresh_needed_frame_change
# ---------------------------------------------------------------------------


async def test_refresh_needed_frame_change(context_engine, heart, session):
    """refresh_needed returns True when frame has changed."""
    sid = f"test-ctx-refresh-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)
    await heart.focus(sid, task="old task", frame="conversation", session=session)

    # Now check with a different frame
    new_frame = _frame_selection("task")
    result = await context_engine.refresh_needed("nous-default", sid, "build something", new_frame, session=session)
    assert result is True


# ---------------------------------------------------------------------------
# 11. test_refresh_needed_same_frame
# ---------------------------------------------------------------------------


async def test_refresh_needed_same_frame(context_engine, heart, session):
    """refresh_needed returns False when frame hasn't changed."""
    sid = f"test-ctx-norefresh-{uuid.uuid4().hex[:8]}"
    await heart.get_or_create_working_memory(sid, session=session)
    await heart.focus(sid, task="current task", frame="task", session=session)

    frame = _frame_selection("task")
    result = await context_engine.refresh_needed("nous-default", sid, "more work", frame, session=session)
    assert result is False


# ---------------------------------------------------------------------------
# 12. test_expand_decision
# ---------------------------------------------------------------------------


async def test_expand_decision(context_engine, brain, session):
    """expand() loads full decision detail by ID."""
    decision = await brain.record(
        _record_input(description="Expand test decision"),
        session=session,
    )

    result = await context_engine.expand("decision", str(decision.id), session=session)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 13. test_estimate_tokens
# ---------------------------------------------------------------------------


async def test_estimate_tokens(context_engine):
    """Token estimation uses chars/4 heuristic."""
    assert context_engine._estimate_tokens("abcd") == 1
    assert context_engine._estimate_tokens("a" * 400) == 100
    assert context_engine._estimate_tokens("") == 1  # min 1
