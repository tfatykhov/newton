"""Bridge extraction: auto-extract structure + function from decision text.

Heuristic-based for now (no LLM calls). Can be upgraded to LLM-based
extraction in a future phase.
"""

from __future__ import annotations

import re

from nous.brain.schemas import BridgeInfo

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MAX_STRUCTURE_LEN = 200


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text, truncated to max length."""
    text = text.strip()
    if not text:
        return ""
    # Split on sentence-ending punctuation followed by whitespace
    parts = _SENTENCE_END.split(text, maxsplit=1)
    sentence = parts[0].strip()
    if len(sentence) > _MAX_STRUCTURE_LEN:
        sentence = sentence[:_MAX_STRUCTURE_LEN].rstrip()
    return sentence


class BridgeExtractor:
    """Extract bridge definitions (structure + function) from decision text."""

    def extract(
        self,
        description: str,
        context: str | None,
        pattern: str | None,
    ) -> BridgeInfo:
        """Extract structure (what it looks like) and function (what it does).

        Heuristics:
        - Structure: first sentence of description (truncated to 200 chars)
        - Function: pattern if available, else first sentence of context
        """
        structure = _first_sentence(description) if description else None

        if pattern:
            function = pattern
        elif context:
            function = _first_sentence(context)
        else:
            function = None

        return BridgeInfo(structure=structure, function=function)
