"""In-process tools for Claude Agent SDK integration.

Provides 4 tool closures that give Claude direct access to Nous memory organs:
- record_decision: Write decisions to Brain
- learn_fact: Store facts in Heart
- recall_deep: Search all memory types (Heart + Brain)
- create_censor: Add guardrails to Heart

Each tool returns MCP-compliant response format and handles errors gracefully.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from nous.brain.brain import Brain
from nous.brain.schemas import ReasonInput, RecordInput
from nous.heart.heart import Heart
from nous.heart.schemas import CensorInput, FactInput

logger = logging.getLogger(__name__)


def create_nous_tools(brain: Brain, heart: Heart) -> dict[str, Any]:
    """Create tool closures with Brain and Heart captured in closure context.

    Returns a dict of async callables suitable for SDK MCP server registration.
    Each closure takes tool parameters, calls Brain/Heart methods, and returns
    MCP-compliant response: {"content": [{"type": "text", "text": "..."}]}.

    All tools are wrapped in try/except to return error messages as tool results.
    """

    async def record_decision(
        description: str,
        confidence: float,
        category: str,
        stakes: str,
        context: str | None = None,
        pattern: str | None = None,
        tags: list[str] | None = None,
        reasons: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Record a decision to the Brain.

        Args:
            description: What was decided
            confidence: 0.0-1.0 confidence level
            category: architecture, process, tooling, security, or integration
            stakes: low, medium, high, or critical
            context: Situation and constraints
            pattern: Abstract pattern this decision represents
            tags: Keywords for filtering
            reasons: List of {type, text} dicts (type: analysis, pattern, empirical, etc.)

        Returns:
            MCP-compliant response with decision ID or error message
        """
        try:
            # Validate and construct input
            reason_inputs = []
            if reasons:
                for r in reasons:
                    if not isinstance(r, dict) or "type" not in r or "text" not in r:
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Error: Invalid reason format. Expected dict with 'type' and 'text', got: {r}"
                                    ),
                                }
                            ]
                        }
                    reason_inputs.append(ReasonInput(type=r["type"], text=r["text"]))

            input_data = RecordInput(
                description=description,
                confidence=confidence,
                category=category,  # type: ignore
                stakes=stakes,  # type: ignore
                context=context,
                pattern=pattern,
                tags=tags or [],
                reasons=reason_inputs,
            )

            # Record to Brain
            result = await brain.record(input_data)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Decision recorded successfully.\n"
                            f"ID: {result.id}\n"
                            f"Quality score: {result.quality_score:.2f}\n"
                            f"Category: {result.category}\n"
                            f"Stakes: {result.stakes}"
                        ),
                    }
                ]
            }

        except Exception as e:
            logger.exception("record_decision tool failed")
            return {"content": [{"type": "text", "text": f"Error recording decision: {e}"}]}

    async def learn_fact(
        content: str,
        category: str | None = None,
        subject: str | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        source_episode_id: str | None = None,
        source_decision_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a fact in the Heart.

        Args:
            content: The fact content
            category: preference, technical, person, tool, concept, or rule
            subject: What/who the fact is about
            confidence: 0.0-1.0 confidence level
            source: Where this fact came from
            source_episode_id: Episode UUID if learned during episode
            source_decision_id: Decision UUID if learned during decision
            tags: Keywords for filtering

        Returns:
            MCP-compliant response with fact ID or error message
        """
        try:
            # Parse UUIDs if provided
            episode_uuid = UUID(source_episode_id) if source_episode_id else None
            decision_uuid = UUID(source_decision_id) if source_decision_id else None

            input_data = FactInput(
                content=content,
                category=category,
                subject=subject,
                confidence=confidence,
                source=source,
                source_episode_id=episode_uuid,
                source_decision_id=decision_uuid,
                tags=tags or [],
            )

            # Store to Heart
            result = await heart.learn(input_data)

            warning_msg = ""
            if result.contradiction_warning:
                warning_msg = (
                    f"\n⚠️ Potential contradiction detected:\n"
                    f"Existing fact: {result.contradiction_warning.existing_content}\n"
                    f"Similarity: {result.contradiction_warning.similarity:.2f}"
                )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Fact learned successfully.\n"
                            f"ID: {result.id}\n"
                            f"Category: {result.category or 'none'}\n"
                            f"Subject: {result.subject or 'none'}"
                            f"{warning_msg}"
                        ),
                    }
                ]
            }

        except ValueError as e:
            # UUID parsing error or validation error
            return {"content": [{"type": "text", "text": f"Validation error: {e}"}]}
        except Exception as e:
            logger.exception("learn_fact tool failed")
            return {"content": [{"type": "text", "text": f"Error learning fact: {e}"}]}

    async def recall_deep(
        query: str,
        limit: int = 10,
        memory_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search across all memory types in Heart and Brain.

        Args:
            query: Search query string
            limit: Maximum results to return
            memory_types: Types to search (episode, fact, procedure, censor, decision)
                         If None or contains "all", searches everything

        Returns:
            MCP-compliant response with ranked results or error message
        """
        try:
            # Determine which types to search
            search_types = memory_types or ["all"]
            search_all = "all" in search_types

            results_text = []

            # Search Heart memory types
            heart_types = []
            if search_all or any(t in search_types for t in ["episode", "fact", "procedure", "censor"]):
                # Determine specific Heart types
                if search_all:
                    heart_types = ["episode", "fact", "procedure", "censor"]
                else:
                    heart_types = [t for t in search_types if t in ["episode", "fact", "procedure", "censor"]]

                if heart_types:
                    heart_results = await heart.recall(query, limit=limit, types=heart_types)
                    if heart_results:
                        results_text.append("=== Heart Memory ===")
                        for i, result in enumerate(heart_results, 1):
                            results_text.append(
                                f"{i}. [{result.type}] {result.summary} (score: {result.score:.3f})"
                            )
                    else:
                        results_text.append("=== Heart Memory ===\nNo results found.")

            # Search Brain decisions
            if search_all or "decision" in search_types:
                decision_results = await brain.query(query, limit=limit)
                if decision_results:
                    results_text.append("\n=== Brain Decisions ===")
                    for i, dec in enumerate(decision_results, 1):
                        score_str = f" (score: {dec.score:.3f})" if dec.score else ""
                        results_text.append(
                            f"{i}. {dec.description} | {dec.category} | {dec.stakes} | "
                            f"confidence: {dec.confidence:.2f}{score_str}"
                        )
                else:
                    results_text.append("\n=== Brain Decisions ===\nNo results found.")

            if not results_text:
                results_text.append("No results found.")

            return {"content": [{"type": "text", "text": "\n".join(results_text)}]}

        except Exception as e:
            logger.exception("recall_deep tool failed")
            return {"content": [{"type": "text", "text": f"Error searching memory: {e}"}]}

    async def create_censor(
        trigger_pattern: str,
        reason: str,
        action: str = "warn",
        domain: str | None = None,
        learned_from_decision: str | None = None,
        learned_from_episode: str | None = None,
    ) -> dict[str, Any]:
        """Create a guardrail censor in the Heart.

        Args:
            trigger_pattern: Pattern to match (substring or regex)
            reason: Why this censor exists
            action: warn, block, or absolute
            domain: Domain this censor applies to (architecture, debugging, etc.)
            learned_from_decision: Decision UUID that triggered this censor
            learned_from_episode: Episode UUID that triggered this censor

        Returns:
            MCP-compliant response with censor ID or error message
        """
        try:
            # Parse UUIDs if provided
            decision_uuid = UUID(learned_from_decision) if learned_from_decision else None
            episode_uuid = UUID(learned_from_episode) if learned_from_episode else None

            input_data = CensorInput(
                trigger_pattern=trigger_pattern,
                reason=reason,
                action=action,  # type: ignore
                domain=domain,
                learned_from_decision=decision_uuid,
                learned_from_episode=episode_uuid,
            )

            # Create censor
            result = await heart.add_censor(input_data)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Censor created successfully.\n"
                            f"ID: {result.id}\n"
                            f"Action: {result.action}\n"
                            f"Domain: {result.domain or 'all'}\n"
                            f"Pattern: {result.trigger_pattern}"
                        ),
                    }
                ]
            }

        except ValueError as e:
            # UUID parsing error or validation error
            return {"content": [{"type": "text", "text": f"Validation error: {e}"}]}
        except Exception as e:
            logger.exception("create_censor tool failed")
            return {"content": [{"type": "text", "text": f"Error creating censor: {e}"}]}

    return {
        "record_decision": record_decision,
        "learn_fact": learn_fact,
        "recall_deep": recall_deep,
        "create_censor": create_censor,
    }


# ---------------------------------------------------------------------------
# SDK MCP server registration
# ---------------------------------------------------------------------------

# JSON Schema definitions for each tool's input parameters.
# Used by claude-agent-sdk @tool decorator so Claude knows
# what arguments each tool accepts.

_RECORD_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "description": "What was decided"},
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 confidence level",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "category": {
            "type": "string",
            "description": "Decision category",
            "enum": ["architecture", "process", "tooling", "security", "integration"],
        },
        "stakes": {
            "type": "string",
            "description": "Stakes level",
            "enum": ["low", "medium", "high", "critical"],
        },
        "context": {"type": "string", "description": "Situation and constraints"},
        "pattern": {"type": "string", "description": "Abstract pattern this decision represents"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keywords for filtering",
        },
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "analysis",
                            "pattern",
                            "empirical",
                            "authority",
                            "analogy",
                            "intuition",
                            "elimination",
                            "constraint",
                        ],
                    },
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
            "description": "Supporting reasons",
        },
    },
    "required": ["description", "confidence", "category", "stakes"],
}

_LEARN_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "The fact content"},
        "category": {
            "type": "string",
            "description": "Fact category",
            "enum": ["preference", "technical", "person", "tool", "concept", "rule"],
        },
        "subject": {"type": "string", "description": "What/who the fact is about"},
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 confidence level",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 1.0,
        },
        "source": {"type": "string", "description": "Where this fact came from"},
        "source_episode_id": {"type": "string", "description": "Episode UUID"},
        "source_decision_id": {"type": "string", "description": "Decision UUID"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keywords for filtering",
        },
    },
    "required": ["content"],
}

_RECALL_DEEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query string"},
        "limit": {
            "type": "integer",
            "description": "Maximum results to return",
            "default": 10,
            "minimum": 1,
            "maximum": 50,
        },
        "memory_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["all", "episode", "fact", "procedure", "censor", "decision"],
            },
            "description": "Types to search. If omitted or contains 'all', searches everything.",
        },
    },
    "required": ["query"],
}

_CREATE_CENSOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trigger_pattern": {
            "type": "string",
            "description": "Pattern to match (substring or regex)",
        },
        "reason": {"type": "string", "description": "Why this censor exists"},
        "action": {
            "type": "string",
            "description": "Censor action",
            "enum": ["warn", "block", "absolute"],
            "default": "warn",
        },
        "domain": {"type": "string", "description": "Domain this censor applies to"},
        "learned_from_decision": {"type": "string", "description": "Decision UUID"},
        "learned_from_episode": {"type": "string", "description": "Episode UUID"},
    },
    "required": ["trigger_pattern", "reason"],
}


def create_nous_mcp_server(brain: Brain, heart: Heart) -> Any:
    """Create an in-process MCP server with Nous tools for the Claude Agent SDK.

    Wraps the closure-based tools from ``create_nous_tools`` with SDK ``@tool``
    decorators and bundles them into a ``McpSdkServerConfig`` suitable for
    ``ClaudeAgentOptions.mcp_servers``.

    Returns:
        ``McpSdkServerConfig`` dict (``{"type": "sdk", "name": ..., "instance": ...}``).
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    closures = create_nous_tools(brain, heart)

    @tool("record_decision", "Record a decision to the Brain (decision intelligence organ)", _RECORD_DECISION_SCHEMA)
    async def sdk_record_decision(args: dict[str, Any]) -> dict[str, Any]:
        return await closures["record_decision"](**args)

    @tool("learn_fact", "Store a fact in the Heart (memory system)", _LEARN_FACT_SCHEMA)
    async def sdk_learn_fact(args: dict[str, Any]) -> dict[str, Any]:
        return await closures["learn_fact"](**args)

    @tool("recall_deep", "Search across all memory types in Heart and Brain", _RECALL_DEEP_SCHEMA)
    async def sdk_recall_deep(args: dict[str, Any]) -> dict[str, Any]:
        return await closures["recall_deep"](**args)

    @tool("create_censor", "Create a guardrail censor in the Heart", _CREATE_CENSOR_SCHEMA)
    async def sdk_create_censor(args: dict[str, Any]) -> dict[str, Any]:
        return await closures["create_censor"](**args)

    return create_sdk_mcp_server(
        "nous",
        version="0.1.0",
        tools=[sdk_record_decision, sdk_learn_fact, sdk_recall_deep, sdk_create_censor],
    )
