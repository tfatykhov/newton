"""Shared data models for the API layer.

Extracted from runner.py to avoid circular imports with compaction.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nous.cognitive.schemas import TurnContext


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """Tracks a multi-turn conversation."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    turn_contexts: list[TurnContext] = field(default_factory=list)
    summary: str | None = None
    compaction_count: int = 0


@dataclass
class ApiResponse:
    """Parsed response from Anthropic Messages API."""

    content: list[dict[str, Any]]  # Raw content blocks from API
    stop_reason: str  # end_turn, max_tokens, tool_use, stop_sequence
    usage: dict[str, int] | None = None
