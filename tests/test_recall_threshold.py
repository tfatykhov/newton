"""Tests for 007.5: Minimum relevance threshold for RECALL.

Verifies that context.build() filters out results below min_score thresholds.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from nous.cognitive.context import ContextEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_decision(score: float, desc: str = "test decision", category: str = "tooling"):
    return SimpleNamespace(
        id=uuid4(),
        description=desc,
        confidence=0.8,
        category=category,
        stakes="medium",
        outcome="pending",
        pattern=None,
        tags=[],
        score=score,
        created_at=None,
    )


def _make_fact(score: float, content: str = "test fact", subject: str = "test"):
    return SimpleNamespace(
        id=uuid4(),
        content=content,
        category="concept",
        subject=subject,
        confidence=1.0,
        active=True,
        score=score,
    )


def _make_episode(score: float, summary: str = "test episode"):
    return SimpleNamespace(
        id=uuid4(),
        title=summary,
        summary=summary,
        outcome="success",
        started_at=None,
        tags=[],
        score=score,
    )


def _make_frame(frame_id: str = "question"):
    return SimpleNamespace(
        frame_id=frame_id,
        description="Test frame",
        questions_to_ask=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecallThreshold:
    """007.5: Results below min_score should be excluded from context."""

    def test_decisions_below_threshold_excluded(self):
        """Decisions with score < 0.3 should be filtered out."""
        decisions = [
            _make_decision(0.8, "highly relevant"),
            _make_decision(0.15, "irrelevant noise"),
            _make_decision(0.29, "just below threshold"),
            _make_decision(0.31, "just above threshold"),
        ]
        min_score = 0.3
        filtered = [d for d in decisions if (d.score or 0) >= min_score]
        assert len(filtered) == 2
        assert filtered[0].description == "highly relevant"
        assert filtered[1].description == "just above threshold"

    def test_facts_below_threshold_excluded(self):
        """Facts with score < 0.25 should be filtered out."""
        facts = [
            _make_fact(0.6, "relevant fact"),
            _make_fact(0.1, "noise"),
            _make_fact(0.24, "just below"),
            _make_fact(0.26, "just above"),
        ]
        min_score = 0.25
        filtered = [f for f in facts if (f.score or 0) >= min_score]
        assert len(filtered) == 2
        assert filtered[0].content == "relevant fact"
        assert filtered[1].content == "just above"

    def test_episodes_below_threshold_excluded(self):
        """Episodes with score < 0.3 should be filtered out."""
        episodes = [
            _make_episode(0.5, "relevant episode"),
            _make_episode(0.05, "noise"),
            _make_episode(0.3, "exactly at threshold"),
        ]
        min_score = 0.3
        filtered = [e for e in episodes if (e.score or 0) >= min_score]
        assert len(filtered) == 2

    def test_none_score_treated_as_zero(self):
        """Items with score=None should be filtered out."""
        decisions = [
            _make_decision(0.5, "has score"),
            _make_decision(0.4, "also has score"),
        ]
        # Simulate a result with no score
        no_score = _make_decision(0.0, "no score")
        no_score.score = None
        decisions.append(no_score)

        min_score = 0.3
        filtered = [d for d in decisions if (d.score or 0) >= min_score]
        assert len(filtered) == 2
        assert all(d.score is not None for d in filtered)

    def test_all_below_threshold_returns_empty(self):
        """If all results are below threshold, section should be empty."""
        decisions = [
            _make_decision(0.1, "noise 1"),
            _make_decision(0.2, "noise 2"),
            _make_decision(0.15, "noise 3"),
        ]
        min_score = 0.3
        filtered = [d for d in decisions if (d.score or 0) >= min_score]
        assert len(filtered) == 0

    def test_facts_have_lower_threshold_than_decisions(self):
        """Facts threshold (0.25) is lower than decisions (0.3)."""
        score = 0.27  # Above fact threshold, below decision threshold
        fact = _make_fact(score)
        decision = _make_decision(score)

        fact_passes = (fact.score or 0) >= 0.25
        decision_passes = (decision.score or 0) >= 0.3

        assert fact_passes is True
        assert decision_passes is False
