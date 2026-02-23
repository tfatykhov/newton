"""Cognitive Layer — The Nous Loop orchestrator.

Wires Brain and Heart into a thinking loop:
Sense -> Frame -> Recall -> Deliberate -> Act -> Monitor -> Learn.

This is NOT an LLM wrapper. The LLM handles "Act". The Cognitive Layer
handles everything else.
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from nous.brain.brain import Brain
from nous.cognitive.context import ContextEngine
from nous.cognitive.deliberation import DeliberationEngine
from nous.cognitive.frames import FrameEngine
from nous.cognitive.monitor import MonitorEngine
from nous.cognitive.schemas import Assessment, TurnContext, TurnResult
from nous.config import Settings
from nous.heart.heart import Heart
from nous.heart.schemas import EpisodeInput, FactInput

logger = logging.getLogger(__name__)

# P2-9: Reflection parsing regex — case-insensitive, supports markdown bullets
_LEARNED_PATTERN = re.compile(r"^\s*[-*]?\s*learned:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


class CognitiveLayer:
    """The Nous Loop — orchestrates Brain and Heart into cognition.

    Usage:
        cognitive = CognitiveLayer(brain, heart, settings)

        # Before LLM turn
        ctx = await cognitive.pre_turn(agent_id, session_id, user_input)
        # ctx.system_prompt contains full context
        # ctx.decision_id set if deliberation started

        # After LLM turn
        result = TurnResult(response_text=llm_output, tool_results=[...])
        assessment = await cognitive.post_turn(agent_id, session_id, result, ctx)

        # End of conversation
        await cognitive.end_session(agent_id, session_id)
    """

    def __init__(
        self,
        brain: Brain,
        heart: Heart,
        settings: Settings,
        identity_prompt: str = "",
    ) -> None:
        self._brain = brain
        self._heart = heart
        self._settings = settings
        # P1-1: Use brain.db (public), not brain._db
        self._frames = FrameEngine(brain.db, settings)
        self._context = ContextEngine(brain, heart, settings, identity_prompt)
        self._deliberation = DeliberationEngine(brain)
        self._monitor = MonitorEngine(brain, heart, settings)

        # Track active episodes per session.
        # P2-10: Known race condition at await boundaries — two coroutines
        # calling pre_turn for same session_id can create duplicate episodes.
        # Acceptable for v0.1 (single agent, single runtime). Full fix with
        # asyncio.Lock per session can come later.
        self._active_episodes: dict[str, str] = {}  # session_id -> episode_id

    async def pre_turn(
        self,
        agent_id: str,
        session_id: str,
        user_input: str,
        session: AsyncSession | None = None,
    ) -> TurnContext:
        """SENSE -> FRAME -> RECALL -> DELIBERATE — prepare for LLM turn.

        Steps:
        1. SENSE: Receive user_input (passed in)
        2. FRAME: Select cognitive frame via FrameEngine.select()
        3. RECALL: Build context via ContextEngine.build()
        4. DELIBERATE: If frame warrants it (decision/task/debug),
           start deliberation and record decision_id
        5. EPISODE: If no active episode for this session, start one
        6. WORKING MEMORY: Update via Heart.focus()

        Return TurnContext with system_prompt, frame, decision_id, metadata.
        """
        # 2. FRAME — select cognitive frame
        try:
            frame = await self._frames.select(agent_id, user_input, session=session)
        except Exception:
            logger.warning("Frame selection failed, falling back to conversation")
            frame = self._frames._default_selection()

        # 3. RECALL — build context
        system_prompt = ""
        recalled_decision_ids: list[str] = []
        recalled_fact_ids: list[str] = []
        context_token_estimate = 0
        try:
            system_prompt, sections = await self._context.build(
                agent_id, session_id, user_input, frame, session=session
            )
            context_token_estimate = sum(s.token_estimate for s in sections)
        except Exception:
            logger.warning("Context build failed, using identity prompt only")
            system_prompt = self._context._identity_prompt or ""

        # 4. DELIBERATE — start if frame warrants it
        decision_id: str | None = None
        try:
            if await self._deliberation.should_deliberate(frame):
                decision_id = await self._deliberation.start(
                    agent_id, user_input[:200], frame, session=session
                )
        except Exception:
            logger.warning("Deliberation start failed, continuing without decision_id")
            decision_id = None

        # 5. EPISODE — start if no active episode for this session
        if session_id not in self._active_episodes:
            try:
                # P1-5: Construct EpisodeInput pydantic model
                episode_input = EpisodeInput(
                    summary=user_input[:200],
                    frame_used=frame.frame_id,
                    trigger="user_message",
                )
                episode = await self._heart.start_episode(episode_input, session=session)
                self._active_episodes[session_id] = str(episode.id)
            except Exception:
                logger.warning("Failed to start episode for session %s", session_id)

        # 6. WORKING MEMORY — update focus
        # P1-7: Must call get_or_create before focus
        try:
            await self._heart.get_or_create_working_memory(session_id, session=session)
            await self._heart.focus(
                session_id, user_input[:200], frame.frame_id, session=session
            )
        except Exception:
            logger.warning("Failed to update working memory for session %s", session_id)

        # Get active censor patterns for TurnContext
        active_censors: list[str] = []
        try:
            censors = await self._heart.list_censors(session=session)
            active_censors = [c.trigger_pattern for c in censors]
        except Exception:
            pass

        return TurnContext(
            system_prompt=system_prompt,
            frame=frame,
            decision_id=decision_id,
            active_censors=active_censors,
            context_token_estimate=context_token_estimate,
            recalled_decision_ids=recalled_decision_ids,
            recalled_fact_ids=recalled_fact_ids,
        )

    async def post_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_result: TurnResult,
        turn_context: TurnContext,
        session: AsyncSession | None = None,
    ) -> Assessment:
        """ACT (done) -> MONITOR -> LEARN — process LLM output.

        Steps:
        1. MONITOR: Assess the turn via MonitorEngine.assess()
        2. LEARN: Extract lessons via MonitorEngine.learn()
        3. DELIBERATION: If decision_id exists, finalize deliberation
           - Confidence: 0.8 if no errors, 0.5 if tool errors, 0.3 if turn error
        4. Emit "turn_completed" event via Brain.emit_event()

        Return Assessment.
        """
        decision_id = turn_context.decision_id
        episode_id = self._active_episodes.get(session_id)

        # 1. MONITOR — assess
        try:
            assessment = await self._monitor.assess(
                agent_id, session_id, turn_result,
                decision_id=decision_id, session=session,
            )
        except Exception:
            logger.warning("Assessment failed, using default")
            assessment = Assessment(actual=turn_result.response_text[:200])

        # 2. LEARN — extract lessons
        try:
            assessment = await self._monitor.learn(
                agent_id, session_id, assessment, turn_result,
                turn_context.frame, episode_id=episode_id, session=session,
            )
        except Exception:
            logger.warning("Learning failed during post_turn")

        # 3. DELIBERATION — finalize if decision exists
        if decision_id:
            try:
                has_tool_errors = any(tr.error for tr in turn_result.tool_results)
                if turn_result.error is not None:
                    confidence = 0.3
                elif has_tool_errors:
                    confidence = 0.5
                else:
                    confidence = 0.8

                await self._deliberation.finalize(
                    decision_id,
                    description=turn_result.response_text[:200],
                    confidence=confidence,
                    session=session,
                )
            except Exception:
                logger.warning("Failed to finalize deliberation for %s", decision_id)

        # 4. EMIT EVENT — P2-7: delegate to Brain.emit_event()
        try:
            await self._brain.emit_event(
                "turn_completed",
                {
                    "session_id": session_id,
                    "frame": turn_context.frame.frame_id,
                    "surprise_level": assessment.surprise_level,
                    "decision_id": decision_id,
                    "has_errors": turn_result.error is not None,
                },
                session=session,
            )
        except Exception:
            logger.warning("Failed to emit turn_completed event")

        return assessment

    async def end_session(
        self,
        agent_id: str,
        session_id: str,
        reflection: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Clean up session state with optional reflection.

        1. If active episode exists for this session:
           - End it with outcome="completed"
           - If reflection provided, include as episode lessons
        2. If reflection provided, extract facts:
           - Parse for "learned: ..." lines (P2-9)
           - Store each as a fact via Heart.learn() with source="reflection"
           - P1-5: Construct FactInput pydantic model
        3. Remove from self._active_episodes (P3-10: use .pop for safety)
        4. Emit "session_ended" event
        """
        # 1. End active episode
        # P3-10: Use .pop(session_id, None) for safety
        episode_id = self._active_episodes.pop(session_id, None)
        if episode_id:
            try:
                lessons = None
                if reflection:
                    lessons = [reflection[:500]]

                # P1-8: No summary param on end()
                await self._heart.end_episode(
                    UUID(episode_id),
                    outcome="success",
                    lessons_learned=lessons,
                    session=session,
                )
            except Exception:
                logger.warning("Failed to end episode %s", episode_id)

        # 2. Extract facts from reflection
        facts_extracted = 0
        if reflection:
            # P2-9: Parse "learned: X" lines
            matches = _LEARNED_PATTERN.findall(reflection)
            for learned_text in matches:
                learned_text = learned_text.strip()
                if not learned_text:
                    continue
                try:
                    # P1-5: Construct FactInput pydantic model
                    fact_input = FactInput(
                        content=learned_text,
                        source="reflection",
                        category="rule",
                    )
                    await self._heart.learn(fact_input, session=session)
                    facts_extracted += 1
                except Exception:
                    logger.warning("Failed to extract fact from reflection: %s", learned_text[:50])

        # 3. Clean up monitor session censor counts
        self._monitor._session_censor_counts.pop(session_id, None)

        # 4. Emit session_ended event — P2-7: delegate to Brain.emit_event()
        try:
            await self._brain.emit_event(
                "session_ended",
                {
                    "session_id": session_id,
                    "had_reflection": reflection is not None,
                    "facts_extracted": facts_extracted,
                },
                session=session,
            )
        except Exception:
            logger.warning("Failed to emit session_ended event")
