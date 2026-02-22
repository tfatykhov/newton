"""Guardrail evaluation engine.

Evaluates JSONB guardrail conditions against a proposed action context.
Conditions use AND logic: all conditions in a guardrail must match to trigger.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nous.brain.schemas import GuardrailResult
from nous.storage.models import Event, Guardrail

logger = logging.getLogger(__name__)


class GuardrailEngine:
    """Evaluate JSONB guardrail conditions against action context."""

    async def check(
        self,
        session: AsyncSession,
        agent_id: str,
        description: str,
        stakes: str,
        confidence: float,
        tags: list[str] | None = None,
        reasons: list[dict] | None = None,
        pattern: str | None = None,
        quality_score: float | None = None,
    ) -> GuardrailResult:
        """Check all active guardrails for the agent.

        Returns GuardrailResult with allowed/blocked_by/warnings.

        Condition matching rules (JSONB):
        - "stakes": "high"       -> exact match on stakes
        - "confidence_lt": 0.5   -> confidence < value
        - "reason_count_lt": 1   -> len(reasons) < value
        - "quality_lt": 0.5      -> quality_score < value

        All conditions in a guardrail use AND logic (all must match to trigger).
        """
        # Load all active guardrails for this agent
        result = await session.execute(
            select(Guardrail).where(
                Guardrail.agent_id == agent_id,
                Guardrail.active.is_(True),
            )
        )
        guardrails = result.scalars().all()

        blocked_by: list[str] = []
        warnings: list[str] = []

        for guardrail in guardrails:
            if self._matches(
                guardrail.condition,
                stakes=stakes,
                confidence=confidence,
                reasons=reasons or [],
                quality_score=quality_score,
            ):
                # Guardrail triggered — classify by severity
                if guardrail.severity in ("block", "absolute"):
                    blocked_by.append(guardrail.name)
                elif guardrail.severity == "warn":
                    warnings.append(guardrail.name)

                # Increment activation_count at SQL level (P3-3: avoid read-modify-write)
                await session.execute(
                    update(Guardrail)
                    .where(Guardrail.id == guardrail.id)
                    .values(
                        activation_count=Guardrail.activation_count + 1,
                        last_activated=datetime.now(UTC),
                    )
                )

                # Log trigger event
                event_type = (
                    "guardrail_blocked"
                    if guardrail.severity in ("block", "absolute")
                    else "guardrail_warned"
                )
                event = Event(
                    agent_id=agent_id,
                    event_type=event_type,
                    data={
                        "guardrail_name": guardrail.name,
                        "severity": guardrail.severity,
                        "description": description,
                        "stakes": stakes,
                        "confidence": confidence,
                    },
                )
                session.add(event)

        allowed = len(blocked_by) == 0
        return GuardrailResult(
            allowed=allowed, blocked_by=blocked_by, warnings=warnings
        )

    def _matches(
        self,
        condition: dict,
        stakes: str,
        confidence: float,
        reasons: list[dict],
        quality_score: float | None,
    ) -> bool:
        """Evaluate all conditions in a guardrail (AND logic).

        All conditions must match for the guardrail to trigger.
        A guardrail with only unknown condition keys does NOT match.
        """
        any_recognized = False
        for key, value in condition.items():
            if key == "stakes":
                any_recognized = True
                if stakes != value:
                    return False
            elif key == "confidence_lt":
                any_recognized = True
                if confidence >= value:
                    return False
            elif key == "reason_count_lt":
                any_recognized = True
                if len(reasons) >= value:
                    return False
            elif key == "quality_lt":
                any_recognized = True
                if quality_score is not None and quality_score >= value:
                    return False
                # If quality_score is None, treat as matching (no quality = low quality)
                if quality_score is None:
                    continue
            else:
                # Unknown condition key — skip (don't block on unknown conditions)
                logger.warning("Unknown guardrail condition key: %s", key)
                continue

        return any_recognized
