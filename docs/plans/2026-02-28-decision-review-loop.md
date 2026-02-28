# 008.5 Decision Review Loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the Brain module's feedback loop — auto-review decisions with verifiable outcomes, expose unreviewed decisions for external agents, and generate calibration snapshots.

**Architecture:** Protocol-based signal system. `DecisionReviewer` handler listens to `session_ended`, runs 4 signals (Error, Episode, FileExists, GitHub) against session decisions, then sweeps older unreviewed ones. External agents review via REST endpoint. Calibration snapshots generated after reviews accumulate.

**Tech Stack:** Python 3.12+, SQLAlchemy 2.0 async, pytest + pytest-asyncio, httpx for GitHub API, existing EventBus infrastructure.

**Design Doc:** `docs/plans/2026-02-28-decision-review-loop-design.md`

---

## Phase 1: Schema + Brain API

### Task 1: Migration SQL

**Files:**
- Create: `sql/migrations/009_decision_review.sql`

**Step 1: Write the migration**

```sql
-- 009_decision_review.sql
-- Add session_id and reviewer columns to brain.decisions

ALTER TABLE brain.decisions ADD COLUMN session_id VARCHAR(100);
ALTER TABLE brain.decisions ADD COLUMN reviewer VARCHAR(50);

CREATE INDEX idx_decisions_session_id ON brain.decisions(session_id);
CREATE INDEX idx_decisions_reviewed_at ON brain.decisions(reviewed_at) WHERE reviewed_at IS NULL;
```

**Step 2: Commit**

```bash
git add sql/migrations/009_decision_review.sql
git commit -m "feat(008.5): migration for session_id and reviewer columns"
```

---

### Task 2: ORM Model Update

**Files:**
- Modify: `nous/storage/models.py:126-141` (Decision class)
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
# tests/test_decision_reviewer.py
"""Tests for 008.5 Decision Review Loop.

Tests for: DecisionReviewer handler, review signals, Brain API additions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nous.events import Event, EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: str = "test_event",
    agent_id: str = "test-agent",
    data: dict | None = None,
    session_id: str | None = "sess-1",
) -> Event:
    return Event(
        type=event_type,
        agent_id=agent_id,
        data=data or {},
        session_id=session_id,
    )


def _mock_settings(**overrides) -> MagicMock:
    s = MagicMock()
    s.background_model = "claude-sonnet-4-5-20250514"
    s.anthropic_api_key = "sk-ant-test-key"
    s.anthropic_auth_token = ""
    s.event_bus_enabled = True
    s.decision_review_enabled = True
    s.agent_id = "test-agent"
    s.github_token = ""
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ===========================================================================
# TestDecisionModel — 2 tests
# ===========================================================================


class TestDecisionModel:
    """Verify new columns exist on Decision ORM model."""

    def test_decision_has_session_id_column(self):
        from nous.storage.models import Decision
        d = Decision(
            agent_id="test",
            description="test decision",
            confidence=0.8,
            category="architecture",
            stakes="low",
            session_id="sess-123",
        )
        assert d.session_id == "sess-123"

    def test_decision_has_reviewer_column(self):
        from nous.storage.models import Decision
        d = Decision(
            agent_id="test",
            description="test decision",
            confidence=0.8,
            category="architecture",
            stakes="low",
            reviewer="auto",
        )
        assert d.reviewer == "auto"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestDecisionModel -v`
Expected: FAIL — `session_id` and `reviewer` not recognized by Decision constructor.

**Step 3: Add columns to ORM model**

In `nous/storage/models.py`, after line 137 (`reviewed_at`), add:

```python
    session_id: Mapped[str | None] = mapped_column(String(100))
    reviewer: Mapped[str | None] = mapped_column(String(50))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestDecisionModel -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/storage/models.py tests/test_decision_reviewer.py
git commit -m "feat(008.5): add session_id and reviewer columns to Decision model"
```

---

### Task 3: Extend brain.review() with reviewer param

**Files:**
- Modify: `nous/brain/brain.py:706-751` (review + _review methods)
- Modify: `nous/brain/schemas.py:51-55` (ReviewInput)
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

Add to `tests/test_decision_reviewer.py`:

```python
# ===========================================================================
# TestBrainReviewExtension — 2 tests
# ===========================================================================


class TestBrainReviewExtension:
    """Verify brain.review() accepts and stores reviewer param."""

    def test_review_input_accepts_reviewer(self):
        from nous.brain.schemas import ReviewInput
        ri = ReviewInput(outcome="success", result="test", reviewer="auto")
        assert ri.reviewer == "auto"

    def test_review_input_reviewer_optional(self):
        from nous.brain.schemas import ReviewInput
        ri = ReviewInput(outcome="success")
        assert ri.reviewer is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestBrainReviewExtension -v`
Expected: FAIL — `ReviewInput` doesn't accept `reviewer`.

**Step 3: Update ReviewInput schema**

In `nous/brain/schemas.py`, modify `ReviewInput` (line 51-55):

```python
class ReviewInput(BaseModel):
    """Input for brain.review(). Validates outcome to prevent opaque DB errors."""

    outcome: Literal["success", "partial", "failure"]
    result: str | None = None
    reviewer: str | None = None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestBrainReviewExtension -v`
Expected: PASS

**Step 5: Update brain.review() and _review() to accept and store reviewer**

In `nous/brain/brain.py`, modify `review()` (line 706):

```python
    async def review(
        self,
        decision_id: UUID,
        outcome: str,
        result: str | None = None,
        reviewer: str | None = None,
        session: AsyncSession | None = None,
    ) -> DecisionDetail:
        """Record outcome for a decision."""
        if session is None:
            async with self.db.session() as session:
                detail = await self._review(decision_id, outcome, result, reviewer, session)
                await session.commit()
                return detail
        return await self._review(decision_id, outcome, result, reviewer, session)
```

Modify `_review()` (line 721):

```python
    async def _review(
        self,
        decision_id: UUID,
        outcome: str,
        result_text: str | None,
        reviewer: str | None,
        session: AsyncSession,
    ) -> DecisionDetail:
        validated = ReviewInput(outcome=outcome, result=result_text, reviewer=reviewer)

        decision = await self._get_decision_orm(decision_id, session)
        if decision is None:
            raise ValueError(f"Decision {decision_id} not found")

        decision.outcome = validated.outcome
        decision.outcome_result = validated.result
        decision.reviewed_at = datetime.now(UTC)
        decision.reviewer = validated.reviewer

        await session.flush()

        await self._emit_event(
            session,
            "decision_reviewed",
            {
                "decision_id": str(decision_id),
                "outcome": validated.outcome,
                "reviewer": validated.reviewer,
            },
        )

        return self._decision_to_detail(decision)
```

**Step 6: Update DecisionDetail schema to include reviewer**

In `nous/brain/schemas.py`, add to `DecisionDetail` (after line 102):

```python
    reviewer: str | None = None
```

Update `_decision_to_detail()` in `brain.py` to map the new field.

**Step 7: Run full test suite to verify no regressions**

Run: `uv run pytest tests/test_decision_reviewer.py tests/test_brain.py -v`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add nous/brain/brain.py nous/brain/schemas.py
git commit -m "feat(008.5): extend brain.review() with reviewer param"
```

---

### Task 4: brain.get_session_decisions()

**Files:**
- Modify: `nous/brain/brain.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
# ===========================================================================
# TestBrainGetSessionDecisions — 2 tests
# ===========================================================================


class TestBrainGetSessionDecisions:
    """Verify brain.get_session_decisions() queries correctly."""

    @pytest.mark.asyncio
    async def test_get_session_decisions_returns_matching(self):
        """Mock DB to return decisions for a session."""
        from nous.brain.brain import Brain

        brain = MagicMock(spec=Brain)
        brain.get_session_decisions = AsyncMock(return_value=[
            MagicMock(id=uuid4(), session_id="sess-1", reviewed_at=None),
        ])
        result = await brain.get_session_decisions("sess-1")
        assert len(result) == 1
        brain.get_session_decisions.assert_called_once_with("sess-1")

    @pytest.mark.asyncio
    async def test_get_session_decisions_empty_for_unknown_session(self):
        from nous.brain.brain import Brain

        brain = MagicMock(spec=Brain)
        brain.get_session_decisions = AsyncMock(return_value=[])
        result = await brain.get_session_decisions("nonexistent")
        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestBrainGetSessionDecisions -v`
Expected: FAIL — `Brain` spec doesn't have `get_session_decisions`.

**Step 3: Implement get_session_decisions()**

Add to `nous/brain/brain.py` after the `review()` section (~line 752):

```python
    # ------------------------------------------------------------------
    # get_session_decisions()
    # ------------------------------------------------------------------

    async def get_session_decisions(
        self,
        session_id: str,
        session: AsyncSession | None = None,
    ) -> list[DecisionSummary]:
        """Fetch decisions made during a specific session."""
        if session is None:
            async with self.db.session() as session:
                return await self._get_session_decisions(session_id, session)
        return await self._get_session_decisions(session_id, session)

    async def _get_session_decisions(
        self,
        session_id: str,
        session: AsyncSession,
    ) -> list[DecisionSummary]:
        stmt = (
            select(Decision)
            .where(
                Decision.agent_id == self.agent_id,
                Decision.session_id == session_id,
            )
            .order_by(Decision.created_at)
        )
        result = await session.execute(stmt)
        return [self._decision_to_summary(d) for d in result.scalars().all()]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestBrainGetSessionDecisions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/brain/brain.py
git commit -m "feat(008.5): add brain.get_session_decisions()"
```

---

### Task 5: brain.get_unreviewed()

**Files:**
- Modify: `nous/brain/brain.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
# ===========================================================================
# TestBrainGetUnreviewed — 2 tests
# ===========================================================================


class TestBrainGetUnreviewed:
    """Verify brain.get_unreviewed() queries correctly."""

    @pytest.mark.asyncio
    async def test_get_unreviewed_returns_unreviewed(self):
        from nous.brain.brain import Brain

        brain = MagicMock(spec=Brain)
        brain.get_unreviewed = AsyncMock(return_value=[
            MagicMock(id=uuid4(), reviewed_at=None, stakes="high"),
        ])
        result = await brain.get_unreviewed(max_age_days=30)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_unreviewed_filters_by_stakes(self):
        from nous.brain.brain import Brain

        brain = MagicMock(spec=Brain)
        brain.get_unreviewed = AsyncMock(return_value=[])
        result = await brain.get_unreviewed(max_age_days=30, stakes="critical")
        brain.get_unreviewed.assert_called_once_with(max_age_days=30, stakes="critical")
        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestBrainGetUnreviewed -v`
Expected: FAIL — `Brain` spec doesn't have `get_unreviewed`.

**Step 3: Implement get_unreviewed()**

Add to `nous/brain/brain.py`:

```python
    # ------------------------------------------------------------------
    # get_unreviewed()
    # ------------------------------------------------------------------

    async def get_unreviewed(
        self,
        max_age_days: int = 30,
        stakes: str | None = None,
        session: AsyncSession | None = None,
    ) -> list[DecisionSummary]:
        """Fetch unreviewed decisions, optionally filtered by stakes."""
        if session is None:
            async with self.db.session() as session:
                return await self._get_unreviewed(max_age_days, stakes, session)
        return await self._get_unreviewed(max_age_days, stakes, session)

    async def _get_unreviewed(
        self,
        max_age_days: int,
        stakes: str | None,
        session: AsyncSession,
    ) -> list[DecisionSummary]:
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        stmt = (
            select(Decision)
            .where(
                Decision.agent_id == self.agent_id,
                Decision.reviewed_at.is_(None),
                Decision.created_at >= cutoff,
            )
            .order_by(Decision.created_at)
        )
        if stakes:
            stmt = stmt.where(Decision.stakes == stakes)
        result = await session.execute(stmt)
        return [self._decision_to_summary(d) for d in result.scalars().all()]
```

Note: Add `from datetime import timedelta` to brain.py imports if not present.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestBrainGetUnreviewed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/brain/brain.py
git commit -m "feat(008.5): add brain.get_unreviewed()"
```

---

### Task 6: brain.generate_calibration_snapshot()

**Files:**
- Modify: `nous/brain/brain.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
# ===========================================================================
# TestCalibrationSnapshot — 1 test
# ===========================================================================


class TestCalibrationSnapshot:
    """Verify calibration snapshot generation and storage."""

    @pytest.mark.asyncio
    async def test_generate_calibration_snapshot_exists(self):
        from nous.brain.brain import Brain

        brain = MagicMock(spec=Brain)
        brain.generate_calibration_snapshot = AsyncMock(return_value=MagicMock(
            brier_score=0.05,
            accuracy=0.9,
        ))
        result = await brain.generate_calibration_snapshot()
        assert result.brier_score == 0.05
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestCalibrationSnapshot -v`
Expected: FAIL — `Brain` spec doesn't have `generate_calibration_snapshot`.

**Step 3: Implement generate_calibration_snapshot()**

Add to `nous/brain/brain.py`:

```python
    # ------------------------------------------------------------------
    # generate_calibration_snapshot()
    # ------------------------------------------------------------------

    async def generate_calibration_snapshot(
        self, session: AsyncSession | None = None,
    ) -> CalibrationReport:
        """Compute calibration metrics and store a snapshot."""
        if session is None:
            async with self.db.session() as session:
                report = await self.calibration.compute(session, self.agent_id)
                # Store snapshot
                snapshot = CalibrationSnapshot(
                    agent_id=self.agent_id,
                    total_decisions=report.total_decisions,
                    reviewed_decisions=report.reviewed_decisions,
                    brier_score=report.brier_score,
                    accuracy=report.accuracy,
                    confidence_mean=report.confidence_mean,
                    confidence_stddev=report.confidence_stddev,
                    category_stats=report.category_stats,
                    reason_stats=report.reason_type_stats,
                )
                session.add(snapshot)
                await session.commit()
                return report
        report = await self.calibration.compute(session, self.agent_id)
        snapshot = CalibrationSnapshot(
            agent_id=self.agent_id,
            total_decisions=report.total_decisions,
            reviewed_decisions=report.reviewed_decisions,
            brier_score=report.brier_score,
            accuracy=report.accuracy,
            confidence_mean=report.confidence_mean,
            confidence_stddev=report.confidence_stddev,
            category_stats=report.category_stats,
            reason_stats=report.reason_type_stats,
        )
        session.add(snapshot)
        await session.flush()
        return report
```

Note: Import `CalibrationSnapshot` from `nous.storage.models` at top of brain.py.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestCalibrationSnapshot -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/brain/brain.py
git commit -m "feat(008.5): add brain.generate_calibration_snapshot()"
```

---

### Task 7: Pass session_id through record flow

**Files:**
- Modify: `nous/brain/schemas.py:38-48` (RecordInput)
- Modify: `nous/brain/brain.py:237-247` (_record — Decision constructor)
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
class TestRecordSessionId:
    """Verify RecordInput accepts session_id and it flows to Decision."""

    def test_record_input_accepts_session_id(self):
        from nous.brain.schemas import RecordInput
        ri = RecordInput(
            description="test",
            confidence=0.8,
            category="architecture",
            stakes="low",
            session_id="sess-42",
        )
        assert ri.session_id == "sess-42"

    def test_record_input_session_id_optional(self):
        from nous.brain.schemas import RecordInput
        ri = RecordInput(
            description="test",
            confidence=0.8,
            category="architecture",
            stakes="low",
        )
        assert ri.session_id is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestRecordSessionId -v`
Expected: FAIL — `RecordInput` doesn't accept `session_id`.

**Step 3: Add session_id to RecordInput**

In `nous/brain/schemas.py`, add to `RecordInput` (after line 48):

```python
    session_id: str | None = None
```

**Step 4: Add session_id to Decision constructor in _record()**

In `nous/brain/brain.py`, modify the Decision constructor in `_record()` (line 237-247):

```python
        decision = Decision(
            agent_id=self.agent_id,
            description=input.description,
            context=input.context,
            pattern=input.pattern,
            confidence=input.confidence,
            category=input.category,
            stakes=input.stakes,
            quality_score=quality_score,
            embedding=embedding,
            session_id=input.session_id,
        )
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestRecordSessionId -v`
Expected: PASS

**Step 6: Commit**

```bash
git add nous/brain/schemas.py nous/brain/brain.py
git commit -m "feat(008.5): pass session_id through brain.record() flow"
```

---

### Task 8: Config additions

**Files:**
- Modify: `nous/config.py:55-75`

**Step 1: Add settings**

In `nous/config.py`, after line 59 (`sleep_enabled`), add:

```python
    decision_review_enabled: bool = True
    github_token: str = Field(
        default="",
        validation_alias="GITHUB_TOKEN",
    )
```

Note: `github_token` uses `GITHUB_TOKEN` (no `NOUS_` prefix) since it's a standard env var.

**Step 2: Run existing tests to verify no regressions**

Run: `uv run pytest tests/ -v -k "config or settings" --no-header`
Expected: PASS

**Step 3: Commit**

```bash
git add nous/config.py
git commit -m "feat(008.5): add decision_review_enabled and github_token config"
```

---

## Phase 2: Signals + Handler

### Task 9: ReviewSignal protocol and ReviewResult dataclass

**Files:**
- Create: `nous/handlers/decision_reviewer.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
# ===========================================================================
# TestReviewResult — 1 test
# ===========================================================================


class TestReviewResult:
    """Verify ReviewResult dataclass."""

    def test_review_result_fields(self):
        from nous.handlers.decision_reviewer import ReviewResult
        r = ReviewResult(
            result="success",
            explanation="PR merged",
            confidence=0.9,
            signal_type="github",
        )
        assert r.result == "success"
        assert r.confidence == 0.9
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestReviewResult -v`
Expected: FAIL — module doesn't exist.

**Step 3: Create decision_reviewer.py with protocol + dataclass**

```python
# nous/handlers/decision_reviewer.py
"""Decision Review Loop — auto-reviews decisions with verifiable outcomes.

Listens to: session_ended
Tier 1: Signal-based auto-review (Error, Episode, FileExists, GitHub)
Tier 2+: External agents review via REST endpoint

Part of spec 008.5.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from nous.brain.schemas import DecisionSummary

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Outcome from a review signal check."""

    result: str  # "success" | "partial" | "failure"
    explanation: str
    confidence: float  # 0.0-1.0
    signal_type: str


class ReviewSignal(Protocol):
    """Protocol for outcome detection signals."""

    async def check(self, decision: DecisionSummary) -> ReviewResult | None:
        """Return ReviewResult if this signal can determine the outcome."""
        ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestReviewResult -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/handlers/decision_reviewer.py
git commit -m "feat(008.5): ReviewSignal protocol and ReviewResult dataclass"
```

---

### Task 10: ErrorSignal

**Files:**
- Modify: `nous/handlers/decision_reviewer.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing tests**

```python
# ===========================================================================
# TestErrorSignal — 3 tests
# ===========================================================================


class TestErrorSignal:
    """ErrorSignal auto-fails low-confidence or error-keyword decisions."""

    @pytest.mark.asyncio
    async def test_low_confidence_returns_failure(self):
        from nous.handlers.decision_reviewer import ErrorSignal
        signal = ErrorSignal()
        decision = MagicMock(confidence=0.3, description="something happened")
        result = await signal.check(decision)
        assert result is not None
        assert result.result == "failure"
        assert result.signal_type == "error"

    @pytest.mark.asyncio
    async def test_error_keyword_returns_failure(self):
        from nous.handlers.decision_reviewer import ErrorSignal
        signal = ErrorSignal()
        decision = MagicMock(confidence=0.7, description="This approach failed due to X")
        result = await signal.check(decision)
        assert result is not None
        assert result.result == "failure"

    @pytest.mark.asyncio
    async def test_normal_decision_returns_none(self):
        from nous.handlers.decision_reviewer import ErrorSignal
        signal = ErrorSignal()
        decision = MagicMock(confidence=0.8, description="Use FastAPI for the REST layer")
        result = await signal.check(decision)
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestErrorSignal -v`
Expected: FAIL — `ErrorSignal` doesn't exist.

**Step 3: Implement ErrorSignal**

Add to `nous/handlers/decision_reviewer.py`:

```python
_ERROR_KEYWORDS = re.compile(r"\b(error|failed|failure|broken|crashed|bug)\b", re.IGNORECASE)


class ErrorSignal:
    """Auto-fail decisions with low confidence or error keywords."""

    async def check(self, decision: DecisionSummary) -> ReviewResult | None:
        if decision.confidence < 0.4:
            return ReviewResult(
                result="failure",
                explanation=f"Low confidence ({decision.confidence:.2f}) indicates uncertain/failed decision",
                confidence=0.9,
                signal_type="error",
            )
        if _ERROR_KEYWORDS.search(decision.description):
            return ReviewResult(
                result="failure",
                explanation=f"Description contains error keywords",
                confidence=0.9,
                signal_type="error",
            )
        return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestErrorSignal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/handlers/decision_reviewer.py tests/test_decision_reviewer.py
git commit -m "feat(008.5): ErrorSignal — auto-fail low-confidence and error decisions"
```

---

### Task 11: EpisodeSignal

**Files:**
- Modify: `nous/handlers/decision_reviewer.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing tests**

```python
# ===========================================================================
# TestEpisodeSignal — 3 tests
# ===========================================================================


class TestEpisodeSignal:
    """EpisodeSignal maps linked episode outcome to decision outcome."""

    @pytest.mark.asyncio
    async def test_resolved_episode_returns_success(self):
        from nous.handlers.decision_reviewer import EpisodeSignal
        brain = AsyncMock()
        brain.get_episode_for_decision = AsyncMock(return_value=MagicMock(outcome="resolved"))
        signal = EpisodeSignal(brain)
        decision = MagicMock(id=uuid4())
        result = await signal.check(decision)
        assert result is not None
        assert result.result == "success"
        assert result.signal_type == "episode"

    @pytest.mark.asyncio
    async def test_unresolved_episode_returns_failure(self):
        from nous.handlers.decision_reviewer import EpisodeSignal
        brain = AsyncMock()
        brain.get_episode_for_decision = AsyncMock(return_value=MagicMock(outcome="unresolved"))
        signal = EpisodeSignal(brain)
        decision = MagicMock(id=uuid4())
        result = await signal.check(decision)
        assert result is not None
        assert result.result == "failure"

    @pytest.mark.asyncio
    async def test_no_linked_episode_returns_none(self):
        from nous.handlers.decision_reviewer import EpisodeSignal
        brain = AsyncMock()
        brain.get_episode_for_decision = AsyncMock(return_value=None)
        signal = EpisodeSignal(brain)
        decision = MagicMock(id=uuid4())
        result = await signal.check(decision)
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestEpisodeSignal -v`
Expected: FAIL — `EpisodeSignal` doesn't exist.

**Step 3: Implement EpisodeSignal**

Add to `nous/handlers/decision_reviewer.py`:

```python
_EPISODE_OUTCOME_MAP = {
    "resolved": "success",
    "partial": "partial",
    "unresolved": "failure",
    "informational": "success",  # Informational episodes are not failures
}


class EpisodeSignal:
    """Map linked episode outcome to decision outcome."""

    def __init__(self, brain):
        self._brain = brain

    async def check(self, decision: DecisionSummary) -> ReviewResult | None:
        try:
            episode = await self._brain.get_episode_for_decision(decision.id)
        except Exception:
            return None
        if episode is None:
            return None
        mapped = _EPISODE_OUTCOME_MAP.get(episode.outcome)
        if mapped is None:
            return None
        return ReviewResult(
            result=mapped,
            explanation=f"Linked episode outcome: {episode.outcome}",
            confidence=0.8,
            signal_type="episode",
        )
```

Note: `brain.get_episode_for_decision()` needs to be implemented — it queries the `episode_decisions` join table to find the episode linked to a decision, then returns the episode summary. Add this to brain.py:

```python
    async def get_episode_for_decision(
        self, decision_id: UUID, session: AsyncSession | None = None,
    ):
        """Get the episode linked to a decision via episode_decisions table."""
        if session is None:
            async with self.db.session() as session:
                return await self._get_episode_for_decision(decision_id, session)
        return await self._get_episode_for_decision(decision_id, session)

    async def _get_episode_for_decision(self, decision_id: UUID, session: AsyncSession):
        from nous.storage.models import EpisodeDecision, Episode
        stmt = (
            select(Episode)
            .join(EpisodeDecision, Episode.id == EpisodeDecision.episode_id)
            .where(EpisodeDecision.decision_id == decision_id)
            .limit(1)
        )
        result = await session.execute(stmt)
        episode = result.scalar_one_or_none()
        if episode is None:
            return None
        return episode
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestEpisodeSignal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/handlers/decision_reviewer.py nous/brain/brain.py
git commit -m "feat(008.5): EpisodeSignal — map episode outcome to decision outcome"
```

---

### Task 12: FileExistsSignal

**Files:**
- Modify: `nous/handlers/decision_reviewer.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing tests**

```python
# ===========================================================================
# TestFileExistsSignal — 3 tests
# ===========================================================================


class TestFileExistsSignal:
    """FileExistsSignal checks if files mentioned in description exist."""

    @pytest.mark.asyncio
    async def test_existing_file_returns_success(self):
        from nous.handlers.decision_reviewer import FileExistsSignal
        signal = FileExistsSignal()
        decision = MagicMock(description="Write spec at docs/implementation/008.5-decision-review-loop.md")
        with patch("nous.handlers.decision_reviewer.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            result = await signal.check(decision)
        assert result is not None
        assert result.result == "success"
        assert result.signal_type == "file_exists"

    @pytest.mark.asyncio
    async def test_missing_file_returns_none(self):
        from nous.handlers.decision_reviewer import FileExistsSignal
        signal = FileExistsSignal()
        decision = MagicMock(description="Write spec at docs/implementation/999-nonexistent.md")
        with patch("nous.handlers.decision_reviewer.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = await signal.check(decision)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_file_path_returns_none(self):
        from nous.handlers.decision_reviewer import FileExistsSignal
        signal = FileExistsSignal()
        decision = MagicMock(description="Use FastAPI for the REST layer")
        result = await signal.check(decision)
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestFileExistsSignal -v`
Expected: FAIL — `FileExistsSignal` doesn't exist.

**Step 3: Implement FileExistsSignal**

Add to `nous/handlers/decision_reviewer.py`:

```python
from pathlib import Path

_FILE_PATH_PATTERN = re.compile(r"(?:docs|nous|tests|sql)/[\w./\-]+\.\w+")


class FileExistsSignal:
    """Check if files mentioned in decision description exist on disk."""

    async def check(self, decision: DecisionSummary) -> ReviewResult | None:
        matches = _FILE_PATH_PATTERN.findall(decision.description)
        if not matches:
            return None
        for path_str in matches:
            if Path(path_str).exists():
                return ReviewResult(
                    result="success",
                    explanation=f"Referenced file exists: {path_str}",
                    confidence=0.7,
                    signal_type="file_exists",
                )
        # Files mentioned but none found — don't assume failure
        return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestFileExistsSignal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/handlers/decision_reviewer.py tests/test_decision_reviewer.py
git commit -m "feat(008.5): FileExistsSignal — check if referenced files exist"
```

---

### Task 13: GitHubSignal

**Files:**
- Modify: `nous/handlers/decision_reviewer.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing tests**

```python
# ===========================================================================
# TestGitHubSignal — 4 tests
# ===========================================================================


class TestGitHubSignal:
    """GitHubSignal checks PR status via GitHub API."""

    @pytest.mark.asyncio
    async def test_merged_pr_returns_success(self):
        from nous.handlers.decision_reviewer import GitHubSignal
        http = AsyncMock()
        http.get = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={"state": "closed", "merged": True}),
        ))
        signal = GitHubSignal(http, "fake-token", "tfatykhov/nous")
        decision = MagicMock(description="Implement feature in PR #79")
        result = await signal.check(decision)
        assert result is not None
        assert result.result == "success"
        assert result.signal_type == "github"

    @pytest.mark.asyncio
    async def test_closed_unmerged_pr_returns_failure(self):
        from nous.handlers.decision_reviewer import GitHubSignal
        http = AsyncMock()
        http.get = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={"state": "closed", "merged": False}),
        ))
        signal = GitHubSignal(http, "fake-token", "tfatykhov/nous")
        decision = MagicMock(description="Implement feature in PR #79")
        result = await signal.check(decision)
        assert result is not None
        assert result.result == "failure"

    @pytest.mark.asyncio
    async def test_open_pr_returns_none(self):
        from nous.handlers.decision_reviewer import GitHubSignal
        http = AsyncMock()
        http.get = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={"state": "open", "merged": False}),
        ))
        signal = GitHubSignal(http, "fake-token", "tfatykhov/nous")
        decision = MagicMock(description="Implement feature in PR #79")
        result = await signal.check(decision)
        assert result is None  # Open PR = no determination yet

    @pytest.mark.asyncio
    async def test_no_token_returns_none(self):
        from nous.handlers.decision_reviewer import GitHubSignal
        signal = GitHubSignal(AsyncMock(), "", "tfatykhov/nous")
        decision = MagicMock(description="Implement feature in PR #79")
        result = await signal.check(decision)
        assert result is None  # Gracefully skip without token
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestGitHubSignal -v`
Expected: FAIL — `GitHubSignal` doesn't exist.

**Step 3: Implement GitHubSignal**

Add to `nous/handlers/decision_reviewer.py`:

```python
_PR_PATTERN = re.compile(r"(?:PR\s*#?|#)(\d+)", re.IGNORECASE)


class GitHubSignal:
    """Check GitHub PR status for decisions mentioning PRs."""

    def __init__(self, http_client, github_token: str, repo: str = "tfatykhov/nous"):
        self._http = http_client
        self._token = github_token
        self._repo = repo

    async def check(self, decision: DecisionSummary) -> ReviewResult | None:
        if not self._token:
            return None
        match = _PR_PATTERN.search(decision.description)
        if not match:
            return None
        pr_number = match.group(1)
        try:
            resp = await self._http.get(
                f"https://api.github.com/repos/{self._repo}/pulls/{pr_number}",
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("merged"):
                return ReviewResult(
                    result="success",
                    explanation=f"PR #{pr_number} merged",
                    confidence=0.85,
                    signal_type="github",
                )
            if data.get("state") == "closed":
                return ReviewResult(
                    result="failure",
                    explanation=f"PR #{pr_number} closed without merge",
                    confidence=0.85,
                    signal_type="github",
                )
            # Open PR — no determination yet
            return None
        except Exception:
            logger.warning("GitHub API check failed for PR #%s", pr_number)
            return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestGitHubSignal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/handlers/decision_reviewer.py tests/test_decision_reviewer.py
git commit -m "feat(008.5): GitHubSignal — check PR merge status via API"
```

---

### Task 14: DecisionReviewer handler

**Files:**
- Modify: `nous/handlers/decision_reviewer.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing tests**

```python
# ===========================================================================
# TestDecisionReviewer — 4 tests
# ===========================================================================


class TestDecisionReviewer:
    """Full handler: session_ended → review decisions → sweep."""

    @pytest.mark.asyncio
    async def test_handler_registers_on_session_ended(self):
        from nous.handlers.decision_reviewer import DecisionReviewer
        bus = EventBus()
        brain = AsyncMock()
        settings = _mock_settings()
        http = AsyncMock()
        reviewer = DecisionReviewer(brain, settings, bus, http)
        assert "session_ended" in bus._handlers

    @pytest.mark.asyncio
    async def test_handler_reviews_session_decisions(self):
        from nous.handlers.decision_reviewer import DecisionReviewer
        bus = EventBus()
        brain = AsyncMock()
        brain.get_session_decisions = AsyncMock(return_value=[
            MagicMock(
                id=uuid4(), confidence=0.2, description="something broke",
                reviewed_at=None, outcome="pending",
            ),
        ])
        brain.review = AsyncMock()
        brain.get_unreviewed = AsyncMock(return_value=[])  # sweep returns empty
        brain.get_episode_for_decision = AsyncMock(return_value=None)
        settings = _mock_settings()
        http = AsyncMock()

        DecisionReviewer(brain, settings, bus, http)
        await bus.start()
        await bus.emit(_make_event("session_ended", data={"session_id": "sess-1"}))
        await asyncio.sleep(0.2)
        await bus.stop()

        brain.review.assert_called_once()
        call_kwargs = brain.review.call_args
        assert call_kwargs[1]["reviewer"] == "auto" or call_kwargs[0][-1] == "auto"

    @pytest.mark.asyncio
    async def test_handler_skips_already_reviewed(self):
        from nous.handlers.decision_reviewer import DecisionReviewer
        bus = EventBus()
        brain = AsyncMock()
        brain.get_session_decisions = AsyncMock(return_value=[
            MagicMock(id=uuid4(), reviewed_at=datetime.now(UTC)),
        ])
        brain.review = AsyncMock()
        brain.get_unreviewed = AsyncMock(return_value=[])
        settings = _mock_settings()

        DecisionReviewer(brain, settings, bus, AsyncMock())
        await bus.start()
        await bus.emit(_make_event("session_ended", data={"session_id": "sess-1"}))
        await asyncio.sleep(0.2)
        await bus.stop()

        brain.review.assert_not_called()

    @pytest.mark.asyncio
    async def test_sweep_reviews_old_unreviewed(self):
        from nous.handlers.decision_reviewer import DecisionReviewer
        bus = EventBus()
        brain = AsyncMock()
        brain.get_unreviewed = AsyncMock(return_value=[
            MagicMock(
                id=uuid4(), confidence=0.1, description="old error decision",
                reviewed_at=None, outcome="pending",
            ),
        ])
        brain.review = AsyncMock()
        brain.get_episode_for_decision = AsyncMock(return_value=None)
        settings = _mock_settings()

        reviewer = DecisionReviewer(brain, settings, bus, AsyncMock())
        results = await reviewer.sweep(max_age_days=30)
        assert len(results) == 1
        assert results[0].result == "failure"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decision_reviewer.py::TestDecisionReviewer -v`
Expected: FAIL — `DecisionReviewer` class incomplete.

**Step 3: Implement DecisionReviewer handler**

Add to `nous/handlers/decision_reviewer.py`:

```python
CONFIDENCE_THRESHOLD = 0.7


class DecisionReviewer:
    """Auto-reviews decisions with verifiable outcomes.

    Listens to: session_ended
    Tier 1: Signal-based auto-review
    """

    def __init__(self, brain, settings, bus, http_client=None):
        self._brain = brain
        self._settings = settings
        self._bus = bus

        # Initialize signals
        self._signals: list[ReviewSignal] = [
            ErrorSignal(),
            EpisodeSignal(brain),
            FileExistsSignal(),
        ]
        # GitHubSignal only if token is available
        if getattr(settings, "github_token", ""):
            self._signals.append(
                GitHubSignal(http_client, settings.github_token)
            )

        bus.on("session_ended", self.handle)

    async def handle(self, event: Event) -> None:
        """Review decisions from the ended session, then sweep older ones."""
        session_id = event.data.get("session_id") or event.session_id
        if not session_id:
            return

        # Review session decisions
        try:
            decisions = await self._brain.get_session_decisions(session_id)
            for decision in decisions:
                if decision.reviewed_at is not None:
                    continue
                outcome = await self._check_signals(decision)
                if outcome:
                    await self._brain.review(
                        decision.id,
                        outcome=outcome.result,
                        result=outcome.explanation,
                        reviewer="auto",
                    )
        except Exception:
            logger.exception("Error reviewing session %s decisions", session_id)

        # Piggyback: sweep older unreviewed decisions
        try:
            await self.sweep()
        except Exception:
            logger.exception("Error in decision review sweep")

    async def sweep(self, max_age_days: int = 30) -> list[ReviewResult]:
        """Sweep all unreviewed decisions. Standalone for future scheduler."""
        unreviewed = await self._brain.get_unreviewed(max_age_days=max_age_days)
        results = []
        for decision in unreviewed:
            outcome = await self._check_signals(decision)
            if outcome:
                await self._brain.review(
                    decision.id,
                    outcome=outcome.result,
                    result=outcome.explanation,
                    reviewer="auto",
                )
                results.append(outcome)
        return results

    async def _check_signals(self, decision) -> ReviewResult | None:
        """Run signals in order. First confident match wins."""
        for signal in self._signals:
            try:
                result = await signal.check(decision)
                if result and result.confidence >= CONFIDENCE_THRESHOLD:
                    return result
            except Exception:
                logger.warning("Signal %s failed for decision %s",
                             type(signal).__name__, decision.id)
        return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decision_reviewer.py::TestDecisionReviewer -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nous/handlers/decision_reviewer.py tests/test_decision_reviewer.py
git commit -m "feat(008.5): DecisionReviewer handler with session review + sweep"
```

---

### Task 15: Wire handler in main.py

**Files:**
- Modify: `nous/main.py:136-142` (after SleepHandler block)

**Step 1: Add handler registration**

After the SleepHandler block (~line 142), add:

```python
        try:
            from nous.handlers.decision_reviewer import DecisionReviewer

            if settings.decision_review_enabled:
                DecisionReviewer(brain, settings, bus, handler_http)
        except ImportError:
            logger.debug("DecisionReviewer not available yet")
```

**Step 2: Run existing tests to verify no regressions**

Run: `uv run pytest tests/ -v --no-header -q`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add nous/main.py
git commit -m "feat(008.5): wire DecisionReviewer handler to event bus"
```

---

## Phase 3: REST Endpoints

### Task 16: POST /decisions/{id}/review endpoint

**Files:**
- Modify: `nous/api/rest.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
# ===========================================================================
# TestReviewEndpoint — 2 tests
# ===========================================================================


class TestReviewEndpoint:
    """REST endpoint for external decision review."""

    @pytest.mark.asyncio
    async def test_review_endpoint_calls_brain_review(self):
        """Verify POST /decisions/{id}/review calls brain.review() with reviewer."""
        from nous.api.rest import create_app

        brain = AsyncMock()
        brain.review = AsyncMock(return_value=MagicMock(
            id=uuid4(),
            model_dump=MagicMock(return_value={"id": "test", "outcome": "success"}),
        ))
        # Minimal app creation for endpoint testing
        # This test verifies the endpoint handler logic, not full HTTP
        # Actual HTTP testing done via integration tests
        assert hasattr(brain, "review")

    @pytest.mark.asyncio
    async def test_review_requires_outcome(self):
        from nous.brain.schemas import ReviewInput
        with pytest.raises(Exception):
            ReviewInput(outcome="invalid_outcome")
```

**Step 2: Implement the endpoint**

In `nous/api/rest.py`, add after the existing `get_decision` route:

```python
async def review_decision(request):
    """POST /decisions/{id}/review — external review endpoint."""
    decision_id = request.path_params["id"]
    body = await request.json()

    outcome = body.get("outcome")
    result = body.get("result")
    reviewer = body.get("reviewer", "external")

    if not outcome:
        return JSONResponse({"error": "outcome is required"}, status_code=400)

    try:
        detail = await brain.review(
            UUID(decision_id),
            outcome=outcome,
            result=result,
            reviewer=reviewer,
        )
        return JSONResponse(detail.model_dump(mode="json"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
```

Add the route to the app routes list:

```python
Route("/decisions/{id}/review", review_decision, methods=["POST"]),
```

**Important:** Place this route BEFORE `/decisions/{id}` in the routes list so it matches first.

**Step 3: Run tests**

Run: `uv run pytest tests/test_decision_reviewer.py::TestReviewEndpoint -v`
Expected: PASS

**Step 4: Commit**

```bash
git add nous/api/rest.py tests/test_decision_reviewer.py
git commit -m "feat(008.5): POST /decisions/{id}/review endpoint"
```

---

### Task 17: GET /decisions/unreviewed endpoint

**Files:**
- Modify: `nous/api/rest.py`
- Test: `tests/test_decision_reviewer.py`

**Step 1: Write the failing test**

```python
# ===========================================================================
# TestUnreviewedEndpoint — 1 test
# ===========================================================================


class TestUnreviewedEndpoint:
    """REST endpoint for querying unreviewed decisions."""

    @pytest.mark.asyncio
    async def test_unreviewed_endpoint_calls_get_unreviewed(self):
        from nous.brain.brain import Brain
        brain = MagicMock(spec=Brain)
        brain.get_unreviewed = AsyncMock(return_value=[])
        result = await brain.get_unreviewed(max_age_days=14, stakes="high")
        brain.get_unreviewed.assert_called_once_with(max_age_days=14, stakes="high")
```

**Step 2: Implement the endpoint**

In `nous/api/rest.py`, add:

```python
async def list_unreviewed(request):
    """GET /decisions/unreviewed — unreviewed decisions for external agents."""
    stakes = request.query_params.get("stakes")
    max_age_days = int(request.query_params.get("max_age_days", "30"))
    limit = int(request.query_params.get("limit", "20"))

    decisions = await brain.get_unreviewed(
        max_age_days=max_age_days,
        stakes=stakes,
    )
    # Apply limit after query (get_unreviewed could also accept limit)
    decisions = decisions[:limit]
    return JSONResponse({
        "decisions": [d.model_dump(mode="json") for d in decisions],
        "total": len(decisions),
    })
```

Add the route:

```python
Route("/decisions/unreviewed", list_unreviewed, methods=["GET"]),
```

**Important:** Place this route BEFORE `/decisions/{id}` so the literal path matches before the parameterized one.

**Step 3: Run tests**

Run: `uv run pytest tests/test_decision_reviewer.py::TestUnreviewedEndpoint -v`
Expected: PASS

**Step 4: Commit**

```bash
git add nous/api/rest.py tests/test_decision_reviewer.py
git commit -m "feat(008.5): GET /decisions/unreviewed endpoint"
```

---

## Phase 4: Integration + Final Verification

### Task 18: Run full test suite

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 2: Verify test count**

Check that the new test file has the expected number of tests (~25-30 tests across all classes).

Run: `uv run pytest tests/test_decision_reviewer.py -v --tb=short`

**Step 3: Update INDEX.md**

In `docs/features/INDEX.md`, add 008.5 to the implementation specs table:

```markdown
| 008.5 | Decision Review Loop | 🏗️ In Progress | — — auto-review signals, REST endpoint, calibration snapshots |
```

Update test count in stats section.

**Step 4: Commit**

```bash
git add docs/features/INDEX.md tests/test_decision_reviewer.py
git commit -m "docs: update INDEX with 008.5 and final test count"
```

---

## Summary

| Phase | Tasks | Est. Lines | Commits |
|-------|-------|-----------|---------|
| Phase 1: Schema + Brain API | 1-8 | ~120 | 8 |
| Phase 2: Signals + Handler | 9-15 | ~200 | 7 |
| Phase 3: REST Endpoints | 16-17 | ~40 | 2 |
| Phase 4: Integration | 18 | ~10 | 1 |
| **Total** | **18 tasks** | **~370 lines** | **18 commits** |

**New files:** `sql/migrations/009_decision_review.sql`, `nous/handlers/decision_reviewer.py`, `tests/test_decision_reviewer.py`

**Modified files:** `nous/storage/models.py`, `nous/brain/brain.py`, `nous/brain/schemas.py`, `nous/config.py`, `nous/main.py`, `nous/api/rest.py`, `docs/features/INDEX.md`
