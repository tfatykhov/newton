"""Guardrail evaluation engine using CEL expressions.

Evaluates guardrail conditions as CEL expressions against a decision context.
CEL is sandboxed: no I/O, no side effects, deterministic evaluation.

Supports both native CEL expressions and legacy JSONB conditions (auto-converted).
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import celpy
from celpy import celtypes
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nous.brain.schemas import GuardrailResult
from nous.storage.models import Event, Guardrail

logger = logging.getLogger(__name__)

# Shared CEL environment — thread-safe, reusable
_CEL_ENV = celpy.Environment()

# Thread pool for CEL evaluation with timeout protection
_EVAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# CEL evaluation timeout (seconds)
_EVAL_TIMEOUT_SECONDS = 0.1


def _to_cel_value(value: Any) -> Any:
    """Convert Python values to CEL-compatible types."""
    if isinstance(value, bool):
        return celtypes.BoolType(value)
    if isinstance(value, int):
        return celtypes.IntType(value)
    if isinstance(value, float):
        return celtypes.DoubleType(value)
    if isinstance(value, str):
        return celtypes.StringType(value)
    if isinstance(value, list):
        return celtypes.ListType([_to_cel_value(v) for v in value])
    if isinstance(value, dict):
        return celtypes.MapType({celtypes.StringType(k): _to_cel_value(v) for k, v in value.items()})
    if value is None:
        return celtypes.BoolType(False)  # CEL has no null; treat as false
    return celtypes.StringType(str(value))


def _build_activation(
    description: str,
    stakes: str,
    confidence: float,
    category: str | None = None,
    tags: list[str] | None = None,
    reasons: list[dict] | None = None,
    pattern: str | None = None,
    quality_score: float | None = None,
    context: dict | None = None,
) -> dict[str, Any]:
    """Build the CEL activation context as a 'decision' map."""
    # Sanitize context to only include JSON-serializable values
    clean_context = _sanitize_context(context)

    # Convert reasons list with full type+text fields
    reasons_list = reasons or []
    reasons_cel = [
        {
            "type": r.get("type", ""),
            "text": r.get("text", ""),
        }
        for r in reasons_list
    ]

    decision = {
        "description": description,
        "stakes": stakes,
        "confidence": confidence,
        "category": category or "",
        "tags": tags or [],
        "reason_count": len(reasons_list),
        "reasons": reasons_cel,
        "pattern": pattern or "",
        "quality_score": quality_score if quality_score is not None else 0.0,
        "has_pattern": bool(pattern),
        "has_tags": len(tags or []) > 0,
        "context": clean_context,
    }
    return {"decision": _to_cel_value(decision)}


def _sanitize_context(ctx: dict | None) -> dict:
    """Remove non-JSON-serializable values from context."""
    if not ctx:
        return {}

    clean = {}
    for k, v in ctx.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean[k] = v
        elif isinstance(v, list):
            # Recursively sanitize lists
            clean[k] = [item for item in v if isinstance(item, (str, int, float, bool, type(None)))]
        elif isinstance(v, dict):
            # Recursively sanitize dicts
            clean[k] = _sanitize_context(v)
        else:
            logger.warning("Dropping non-JSON context key '%s' (type=%s)", k, type(v).__name__)

    return clean


class GuardrailEngine:
    """Evaluate CEL guardrail expressions against decision context."""

    def __init__(self) -> None:
        # Program cache is managed by lru_cache on _compile_program
        pass

    @lru_cache(maxsize=100)
    def _compile_program(self, expression: str) -> celpy.Runner:
        """Compile and cache a CEL expression (LRU cache with max 100 programs)."""
        try:
            ast = _CEL_ENV.compile(expression)
            return _CEL_ENV.program(ast)
        except Exception:
            logger.error("Failed to compile CEL expression: %s", expression)
            raise

    def validate_expression(self, expression: str) -> tuple[bool, str | None]:
        """Validate CEL expression syntax. Returns (is_valid, error_message)."""
        try:
            self._compile_program(expression)
            return True, None
        except Exception as e:
            return False, str(e)

    def _evaluate(self, expression: str, activation: dict, severity: str = "block") -> bool:
        """Evaluate a CEL expression with timeout and fail-closed for block severity.

        Args:
            expression: CEL expression string
            activation: CEL activation context (decision map)
            severity: Guardrail severity (block, warn, absolute)

        Returns:
            True if condition matches (guardrail triggers), False otherwise
        """
        # Determine fail behavior: block/absolute severity should fail closed
        should_fail_closed = severity in ("block", "absolute")

        try:
            program = self._compile_program(expression)
            future = _EVAL_EXECUTOR.submit(program.evaluate, activation)
            result = future.result(timeout=_EVAL_TIMEOUT_SECONDS)
            return bool(result)
        except concurrent.futures.TimeoutError:
            logger.error("CEL evaluation timed out: %s (severity=%s)", expression, severity)
            return should_fail_closed  # Fail closed on timeout for block severity
        except Exception:
            logger.error("CEL evaluation failed: %s (severity=%s)", expression, severity, exc_info=True)
            return should_fail_closed  # Fail closed on error for block severity

    async def check(
        self,
        session: AsyncSession,
        agent_id: str,
        description: str,
        stakes: str,
        confidence: float,
        category: str | None = None,
        tags: list[str] | None = None,
        reasons: list[dict] | None = None,
        pattern: str | None = None,
        quality_score: float | None = None,
        context: dict | None = None,
    ) -> GuardrailResult:
        """Check all active guardrails for the agent.

        Args:
            context: Arbitrary key-value dict accessible as decision.context in CEL.
                     This is how callers pass custom fields (e.g., architecture_review=true).
        """
        # Load all active guardrails for this agent, ordered by priority ASC
        result = await session.execute(
            select(Guardrail)
            .where(
                Guardrail.agent_id == agent_id,
                Guardrail.active.is_(True),
            )
            .order_by(Guardrail.priority.asc(), Guardrail.created_at.asc())
        )
        guardrails = result.scalars().all()

        activation = _build_activation(
            description=description,
            stakes=stakes,
            confidence=confidence,
            category=category,
            tags=tags,
            reasons=reasons,
            pattern=pattern,
            quality_score=quality_score,
            context=context,
        )

        blocked_by: list[str] = []
        warnings: list[str] = []

        for guardrail in guardrails:
            expression = self._get_expression(guardrail.condition)
            if expression and self._evaluate(expression, activation, guardrail.severity):
                if guardrail.severity in ("block", "absolute"):
                    blocked_by.append(guardrail.name)
                elif guardrail.severity == "warn":
                    warnings.append(guardrail.name)

                # Increment activation count
                await session.execute(
                    update(Guardrail)
                    .where(Guardrail.id == guardrail.id)
                    .values(
                        activation_count=Guardrail.activation_count + 1,
                        last_activated=datetime.now(UTC),
                    )
                )

                # Log trigger event
                event_type = "guardrail_blocked" if guardrail.severity in ("block", "absolute") else "guardrail_warned"
                session.add(
                    Event(
                        agent_id=agent_id,
                        event_type=event_type,
                        data={
                            "guardrail_name": guardrail.name,
                            "severity": guardrail.severity,
                            "expression": expression,
                            "stakes": stakes,
                            "confidence": confidence,
                        },
                    )
                )

        return GuardrailResult(allowed=len(blocked_by) == 0, blocked_by=blocked_by, warnings=warnings)

    def _get_expression(self, condition: dict | str) -> str | None:
        """Extract CEL expression from condition field.

        Supports three formats:
        1. String: direct CEL expression
           "decision.stakes == 'high' && decision.confidence < 0.5"

        2. Dict with 'cel' key: CEL expression in a dict
           {"cel": "decision.stakes == 'high'"}

        3. Legacy JSONB dict: auto-convert to CEL (backward compatible)
           {"stakes": "high", "confidence_lt": 0.5}
           → "decision.stakes == 'high' && decision.confidence < 0.5"
        """
        try:
            if isinstance(condition, str):
                return condition

            if isinstance(condition, dict):
                if "cel" in condition:
                    cel_val = condition["cel"]
                    if isinstance(cel_val, str):
                        return cel_val
                    else:
                        logger.error("CEL value is not string: %s", cel_val)
                        return None
                # Legacy JSONB → CEL conversion
                return self._jsonb_to_cel(condition)

            logger.error("Invalid condition type: %s", type(condition))
            return None
        except Exception:
            logger.error("Failed to parse condition: %s", condition, exc_info=True)
            return None

    def _jsonb_to_cel(self, condition: dict) -> str:
        """Convert legacy JSONB conditions to CEL expressions."""
        parts = []
        for key, value in condition.items():
            if key == "stakes":
                parts.append(f"decision.stakes == '{value}'")
            elif key == "confidence_lt":
                parts.append(f"decision.confidence < {value}")
            elif key == "reason_count_lt":
                parts.append(f"decision.reason_count < {value}")
            elif key == "quality_lt":
                parts.append(f"decision.quality_score < {value}")
            else:
                logger.warning("Unknown legacy condition key: %s (skipping)", key)

        cel_expr = " && ".join(parts) if parts else "false"

        # Log legacy conversion at INFO level (opt-in logging for observability)
        logger.info("Legacy JSONB→CEL conversion: %s → %s", condition, cel_expr)

        return cel_expr
