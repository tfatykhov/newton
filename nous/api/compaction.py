"""Conversation compaction — tool result pruning and history management.

Spec 008.1: Two-layer approach:
  Layer 1: Tool output pruning (per-request, no LLM)
  Layer 2: History compaction (rare, LLM-powered) — Phase 2

This module is independent of AgentRunner to avoid circular imports
and keep runner.py focused on orchestration.
"""

from __future__ import annotations

import logging
from typing import Any

from nous.config import Settings

logger = logging.getLogger(__name__)


class TokenEstimator:
    """Estimates token counts with optional calibration from API usage.

    Starts with chars/4 heuristic. Improves via calibrate() after each
    API response using actual input_tokens from usage data.

    Limitations (acknowledged):
    - Resets on container restart (ephemeral)
    - actual_tokens from API includes system prompt overhead
    - Alpha=0.1 means ~10 samples to ~65% convergence
    - 20K safety margin absorbs estimation error
    """

    def __init__(self) -> None:
        self._ratio: float = 0.25  # tokens per char (chars/4 default)
        self._samples: int = 0

    @property
    def samples(self) -> int:
        """Number of calibration samples received."""
        return self._samples

    @property
    def ratio(self) -> float:
        """Current tokens-per-char ratio."""
        return self._ratio

    def estimate(self, text: str | Any) -> int:
        """Estimate token count for text content."""
        if isinstance(text, str):
            return max(1, int(len(text) * self._ratio))
        return max(1, int(len(str(text)) * self._ratio))

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens for a message list."""
        return sum(self.estimate(m.get("content", "")) + 4 for m in messages)

    def calibrate(self, input_chars: int, actual_tokens: int) -> None:
        """Update ratio from actual API input_tokens. EMA with alpha=0.1."""
        if input_chars <= 0 or actual_tokens <= 0:
            return
        observed = actual_tokens / input_chars
        self._ratio = 0.1 * observed + 0.9 * self._ratio
        self._samples += 1


class ConversationCompactor:
    """Manages tool result pruning (Layer 1) and history compaction (Layer 2).

    Layer 1 (Phase 1): Prunes tool results during tool loops to prevent
    in-turn context window overflow. No LLM calls.

    Layer 2 (Phase 2): Compacts conversation history via structured
    summarization when token budget is exceeded.

    Owns a TokenEstimator instance. AgentRunner accesses it via
    compactor.estimator for calibration after API responses.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.estimator = TokenEstimator()

    # ------------------------------------------------------------------
    # Layer 1: Tool Output Pruning
    # ------------------------------------------------------------------

    @staticmethod
    def is_tool_result_message(msg: dict[str, Any]) -> bool:
        """Check if a message contains tool results (not regular user text).

        Tool result messages have role="user" with content as a list of
        dicts containing type="tool_result". Regular user messages have
        string content.
        """
        content = msg.get("content")
        return (
            msg.get("role") == "user"
            and isinstance(content, list)
            and len(content) > 0
            and isinstance(content[0], dict)
            and content[0].get("type") == "tool_result"
        )

    def prune_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """Prune old tool results from in-turn message accumulation.

        Mutates messages in place. Two-phase approach:
        1. Soft-trim: Keep head + tail of oversized results
        2. Hard-clear: Replace very old results with placeholder

        Never modifies user text messages or assistant content blocks.
        Protects the last keep_last_tool_results tool-result messages.
        """
        if not self._settings.tool_pruning_enabled:
            return

        # Find all tool result message indices
        tool_indices: list[int] = [
            i for i, msg in enumerate(messages)
            if self.is_tool_result_message(msg)
        ]

        if not tool_indices:
            return

        # Protection zone: last N tool result messages
        protected = set(tool_indices[-self._settings.keep_last_tool_results:])

        soft_trimmed = 0
        hard_cleared = 0

        for pos, idx in enumerate(tool_indices):
            if idx in protected:
                continue

            msg = messages[idx]
            content = msg["content"]  # list of tool_result dicts
            age = len(tool_indices) - pos  # distance from end

            # Hard-clear: very old results
            if age > self._settings.tool_hard_clear_after:
                for item in content:
                    if self._has_image_content(item):
                        continue
                    item["content"] = (
                        "[Tool output cleared — content was processed in earlier turns]"
                    )
                hard_cleared += 1
                continue

            # Soft-trim: oversized results
            for item in content:
                if self._has_image_content(item):
                    continue
                text = item.get("content", "")
                if not isinstance(text, str):
                    continue
                if len(text) > self._settings.tool_soft_trim_chars:
                    head = self._settings.tool_soft_trim_head
                    tail = self._settings.tool_soft_trim_tail
                    original_len = len(text)
                    item["content"] = (
                        f"{text[:head]}\n\n"
                        f"--- trimmed (kept {head} head + {tail} tail "
                        f"of {original_len} chars) ---\n\n"
                        f"{text[-tail:]}"
                    )
                    soft_trimmed += 1

        if soft_trimmed or hard_cleared:
            logger.info(
                "Pruned tool results: soft-trimmed=%d, hard-cleared=%d "
                "(total tool msgs=%d, protected=%d)",
                soft_trimmed, hard_cleared,
                len(tool_indices), len(protected),
            )

    @staticmethod
    def _has_image_content(item: dict[str, Any]) -> bool:
        """Check if a tool result item contains image content."""
        content = item.get("content")
        if isinstance(content, list):
            return any(
                isinstance(block, dict) and block.get("type") == "image"
                for block in content
            )
        return False

    # ------------------------------------------------------------------
    # Layer 2: History Compaction (Phase 2 — stubs)
    # ------------------------------------------------------------------

    def should_compact(self, system_tokens: int, history_tokens: int) -> bool:
        """Check if compaction is needed. Phase 2 implementation."""
        if not self._settings.compaction_enabled:
            return False
        total = system_tokens + history_tokens
        return total > self._settings.compaction_threshold

    def find_cut_point(
        self, messages: list[dict[str, Any]], keep_recent_tokens: int
    ) -> int:
        """Find where to split old/recent messages. Phase 2 implementation."""
        accumulated = 0
        for i in range(len(messages) - 1, -1, -1):
            accumulated += self.estimator.estimate(messages[i].get("content", ""))
            if accumulated >= keep_recent_tokens:
                for j in range(i, len(messages)):
                    if messages[j].get("role") == "user":
                        return j
                return 0
        return 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text(content: list[dict[str, Any]]) -> str:
        """Extract text from API response content blocks."""
        return "".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )
