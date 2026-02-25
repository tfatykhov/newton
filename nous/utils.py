"""Shared utility functions for Nous."""

from __future__ import annotations


def text_overlap(a: str, b: str) -> float:
    """Word overlap ratio for deduplication.

    Filters words shorter than 3 characters to avoid false positives
    from stop words (the, is, a, in, etc.).

    Returns 0.0-1.0 representing what fraction of the smaller
    text's words appear in the larger text.
    """
    words_a = set(w for w in a.lower().split() if len(w) >= 3)
    words_b = set(w for w in b.lower().split() if len(w) >= 3)
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    smaller = min(len(words_a), len(words_b))
    return overlap / smaller
