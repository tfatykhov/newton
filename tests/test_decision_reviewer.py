"""Unit tests for 008.5 Decision Review Loop — API surface tests.

Tests cover Tasks 2-8: ORM model updates, schema extensions,
new Brain methods, and config additions.  All tests use mocks
(no real DB) to verify the public contract.

7 test classes, ~10 tests total.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nous.brain.schemas import (
    CalibrationReport,
    DecisionDetail,
    DecisionSummary,
    RecordInput,
    ReviewInput,
)
from nous.config import Settings
from nous.storage.models import Decision


# ---------------------------------------------------------------------------
# Task 2: ORM Model — Decision accepts session_id and reviewer
# ---------------------------------------------------------------------------


class TestDecisionModel:
    """Verify the Decision ORM model accepts the new columns."""

    def test_decision_accepts_session_id(self):
        """Decision constructor should accept session_id."""
        d = Decision(
            agent_id="test",
            description="test decision",
            confidence=0.8,
            category="architecture",
            stakes="medium",
            session_id="sess-123",
        )
        assert d.session_id == "sess-123"

    def test_decision_accepts_reviewer(self):
        """Decision constructor should accept reviewer."""
        d = Decision(
            agent_id="test",
            description="test decision",
            confidence=0.8,
            category="architecture",
            stakes="medium",
            reviewer="auto-reviewer",
        )
        assert d.reviewer == "auto-reviewer"


# ---------------------------------------------------------------------------
# Task 3: ReviewInput accepts reviewer
# ---------------------------------------------------------------------------


class TestBrainReviewExtension:
    """Verify ReviewInput schema accepts the reviewer field."""

    def test_review_input_accepts_reviewer(self):
        """ReviewInput should accept an optional reviewer field."""
        ri = ReviewInput(outcome="success", result="it worked", reviewer="auto")
        assert ri.reviewer == "auto"
        assert ri.outcome == "success"

    def test_review_input_reviewer_optional(self):
        """ReviewInput reviewer should default to None."""
        ri = ReviewInput(outcome="failure")
        assert ri.reviewer is None


# ---------------------------------------------------------------------------
# Task 3b: DecisionDetail includes reviewer
# ---------------------------------------------------------------------------


class TestDecisionDetailReviewer:
    """Verify DecisionDetail schema includes reviewer."""

    def test_decision_detail_includes_reviewer(self):
        """DecisionDetail should have a reviewer field."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        dd = DecisionDetail(
            id="00000000-0000-0000-0000-000000000001",
            agent_id="test",
            description="test",
            confidence=0.8,
            category="architecture",
            stakes="medium",
            outcome="success",
            reviewed_at=now,
            reviewer="auto-reviewer",
            created_at=now,
            updated_at=now,
        )
        assert dd.reviewer == "auto-reviewer"


# ---------------------------------------------------------------------------
# Task 7: RecordInput accepts session_id
# ---------------------------------------------------------------------------


class TestRecordSessionId:
    """Verify RecordInput schema accepts session_id."""

    def test_record_input_accepts_session_id(self):
        """RecordInput should accept an optional session_id field."""
        ri = RecordInput(
            description="Use Postgres",
            confidence=0.9,
            category="architecture",
            stakes="high",
            session_id="sess-abc",
        )
        assert ri.session_id == "sess-abc"

    def test_record_input_session_id_optional(self):
        """RecordInput session_id should default to None."""
        ri = RecordInput(
            description="Use Postgres",
            confidence=0.9,
            category="architecture",
            stakes="high",
        )
        assert ri.session_id is None


# ---------------------------------------------------------------------------
# Task 4 + 5: get_session_decisions / get_unreviewed exist on Brain
# ---------------------------------------------------------------------------


class TestBrainNewMethods:
    """Verify new methods exist on the Brain class."""

    def test_get_session_decisions_exists(self):
        """Brain should have a get_session_decisions method."""
        from nous.brain.brain import Brain

        assert hasattr(Brain, "get_session_decisions")
        assert callable(getattr(Brain, "get_session_decisions"))

    def test_get_unreviewed_exists(self):
        """Brain should have a get_unreviewed method."""
        from nous.brain.brain import Brain

        assert hasattr(Brain, "get_unreviewed")
        assert callable(getattr(Brain, "get_unreviewed"))


# ---------------------------------------------------------------------------
# Task 6: generate_calibration_snapshot exists on Brain
# ---------------------------------------------------------------------------


class TestCalibrationSnapshot:
    """Verify generate_calibration_snapshot method exists."""

    def test_generate_calibration_snapshot_exists(self):
        """Brain should have a generate_calibration_snapshot method."""
        from nous.brain.brain import Brain

        assert hasattr(Brain, "generate_calibration_snapshot")
        assert callable(getattr(Brain, "generate_calibration_snapshot"))


# ---------------------------------------------------------------------------
# Task 8: Config additions
# ---------------------------------------------------------------------------


class TestConfigAdditions:
    """Verify new config fields exist."""

    def test_decision_review_enabled_default(self):
        """Settings should have decision_review_enabled defaulting to True."""
        s = Settings(ANTHROPIC_API_KEY="test")
        assert s.decision_review_enabled is True

    def test_github_token_default(self):
        """Settings should have github_token defaulting to empty string."""
        s = Settings(ANTHROPIC_API_KEY="test")
        assert s.github_token == ""
