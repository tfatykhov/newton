"""Unit tests for quality scoring logic (pure function, no DB)."""

from nous.brain.quality import QualityScorer, compute_quality_score


def test_full_quality():
    """All fields present -> score = 1.0."""
    score = compute_quality_score(
        tags=["python", "architecture"],
        reasons=[
            {"type": "analysis", "text": "Analyzed trade-offs"},
            {"type": "pattern", "text": "Follows established patterns"},
        ],
        pattern="Use proven technology stacks",
        context="Evaluating database options for the project",
    )
    assert score == 1.0


def test_no_metadata():
    """No tags, reasons, pattern, context -> score = 0.0."""
    score = compute_quality_score(
        tags=[],
        reasons=[],
        pattern=None,
        context=None,
    )
    assert score == 0.0


def test_tags_only():
    """Only tags present -> score = 0.25."""
    score = compute_quality_score(
        tags=["python"],
        reasons=[],
        pattern=None,
        context=None,
    )
    assert score == 0.25


def test_reason_diversity_bonus():
    """2+ reason types -> extra 0.10 bonus on top of base 0.25 for reasons."""
    # One reason type only: tags(0.25) + reasons(0.25) = 0.50
    score_single = compute_quality_score(
        tags=["test"],
        reasons=[
            {"type": "analysis", "text": "Reason 1"},
            {"type": "analysis", "text": "Reason 2"},
        ],
        pattern=None,
        context=None,
    )
    assert score_single == 0.50

    # Two distinct reason types: tags(0.25) + reasons(0.25) + diversity(0.10) = 0.60
    score_diverse = compute_quality_score(
        tags=["test"],
        reasons=[
            {"type": "analysis", "text": "Reason 1"},
            {"type": "pattern", "text": "Reason 2"},
        ],
        pattern=None,
        context=None,
    )
    assert score_diverse == 0.60


def test_quality_threshold():
    """Below 0.5 is 'low quality' — only reasons present = 0.25."""
    score = compute_quality_score(
        tags=[],
        reasons=[{"type": "intuition", "text": "Gut feeling"}],
        pattern=None,
        context=None,
    )
    assert score == 0.25
    assert score < 0.5, "Score should be below the low-quality threshold"


def test_quality_scorer_class():
    """QualityScorer class wrapper delegates to compute_quality_score."""
    scorer = QualityScorer()
    score = scorer.compute(
        tags=["db"],
        reasons=[{"type": "analysis", "text": "Analyzed"}],
        pattern="Use Postgres",
        context="Choosing a database",
    )
    # tags(0.25) + reasons(0.25) + pattern(0.25) + context(0.15) = 0.90
    assert score == 0.90
