"""Deliberation engine — manages the deliberation lifecycle for a turn.

Thin wrapper around Brain.record(), Brain.think(), Brain.update()
providing a clean start -> think -> finalize flow.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from nous.brain.brain import Brain
from nous.brain.schemas import ReasonInput, RecordInput
from nous.cognitive.schemas import FrameSelection

logger = logging.getLogger(__name__)

# Frames that trigger deliberation (D8)
_DELIBERATION_FRAMES = {"decision", "task", "debug"}


class DeliberationEngine:
    """Manages the deliberation lifecycle for a single turn.

    Wraps Brain.record(), Brain.think(), Brain.update() into
    a clean start -> think -> finalize flow.
    """

    def __init__(self, brain: Brain) -> None:
        self._brain = brain

    async def start(
        self,
        agent_id: str,
        description: str,
        frame: FrameSelection,
        session: AsyncSession | None = None,
    ) -> str:
        """Begin deliberation -- record intent, return decision_id as string.

        Creates a decision via Brain.record() with:
        - description: "Plan: {description}"
        - confidence: 0.5 (to be updated in finalize)
        - category: frame.default_category or "process"
        - stakes: frame.default_stakes or "low"
        - tags: [frame.frame_id]
        - reasons: at least 1 ReasonInput (required by guardrail)

        P1-2: Constructs RecordInput pydantic model.
        P2-6: Returns str(uuid) for TurnContext storage.
        """
        # P1-2: Build RecordInput with at least 1 reason
        record_input = RecordInput(
            description=f"Plan: {description}",
            confidence=0.5,
            category=frame.default_category or "process",
            stakes=frame.default_stakes or "low",
            tags=[frame.frame_id],
            reasons=[
                ReasonInput(
                    type="analysis",
                    text=f"Frame '{frame.frame_name}' triggered deliberation for: {description[:100]}",
                )
            ],
        )

        detail = await self._brain.record(record_input, session=session)
        # P2-6: Return as string for TurnContext.decision_id
        return str(detail.id)

    async def think(
        self,
        decision_id: str,
        thought: str,
        agent_id: str,
        session: AsyncSession | None = None,
    ) -> None:
        """Capture a micro-thought during deliberation.

        P2-6: Converts str decision_id to UUID for Brain.think().
        """
        await self._brain.think(UUID(decision_id), thought, session=session)

    async def finalize(
        self,
        decision_id: str,
        description: str,
        confidence: float,
        context: str | None = None,
        pattern: str | None = None,
        tags: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Update decision with final outcome.

        P1-3: Uses extended Brain.update() with confidence param.
        P2-6: Converts str decision_id to UUID for Brain.update().
        Only updates fields that are not None.
        """
        await self._brain.update(
            decision_id=UUID(decision_id),
            description=description,
            context=context,
            pattern=pattern,
            confidence=confidence,
            tags=tags,
            session=session,
        )

    async def delete(
        self,
        decision_id: str,
        session: AsyncSession | None = None,
    ) -> None:
        """Delete a deliberation record for non-decisions.

        Informational responses shouldn't create decision records at all.
        Instead of marking as failure (which pollutes Brain with noise),
        remove the record entirely.
        """
        await self._brain.delete(UUID(decision_id), session=session)

    async def should_deliberate(self, frame: FrameSelection) -> bool:
        """Should this frame trigger deliberation?

        Returns True for: decision, task, debug (D8)
        Returns False for: conversation, question, creative
        """
        return frame.frame_id in _DELIBERATION_FRAMES
