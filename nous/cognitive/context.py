"""Context assembly engine — builds system prompt within token budgets.

Queries Brain and Heart, formats results as markdown sections,
and concatenates them in priority order within per-section token budgets.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from nous.brain.brain import Brain
from nous.cognitive.schemas import ContextBudget, ContextSection, FrameSelection
from nous.config import Settings
from nous.heart.heart import Heart
from nous.heart.search import apply_frame_boost

logger = logging.getLogger(__name__)


class ContextEngine:
    """Assembles context from Brain and Heart within token budgets."""

    CHARS_PER_TOKEN = 4

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
        self._identity_prompt = identity_prompt

    async def build(
        self,
        agent_id: str,
        session_id: str,
        input_text: str,
        frame: FrameSelection,
        session: AsyncSession | None = None,
    ) -> tuple[str, list[ContextSection]]:
        """Build system prompt + context sections within budget.

        Returns (system_prompt_string, sections_list).

        Assembly order (by priority):
        1. Identity prompt (always included, static)
        2. Active censors (always, action=block first)
        3. Frame description + questions_to_ask
        4. Working memory (current task + open threads)
        5. Similar decisions from Brain.query()
        6. Relevant facts from Heart.search_facts()
        7. Relevant procedures from Heart.search_procedures()
        8. Related episodes from Heart.search_episodes()

        Each section is truncated to its budget allocation.
        Sections are skipped entirely if budget is 0 for that layer.
        """
        budget = ContextBudget.for_frame(frame.frame_id)
        sections: list[ContextSection] = []
        _active_censor_names: list[str] = []  # Populated in step 2, used for frame boost

        # 1. Identity (always included)
        if self._identity_prompt:
            identity_text = self._truncate_to_budget(self._identity_prompt, budget.identity)
            sections.append(
                ContextSection(
                    priority=1,
                    label="Identity",
                    content=identity_text,
                    token_estimate=self._estimate_tokens(identity_text),
                )
            )

        # 2. Censors (P2-5: per-section isolation)
        if budget.censors > 0:
            try:
                censors = await self._heart.list_censors(session=session)
                if censors:
                    _active_censor_names = [str(getattr(c, "id", "")) for c in censors]
                    censor_text = self._format_censors(censors)
                    censor_text = self._truncate_to_budget(censor_text, budget.censors)
                    sections.append(
                        ContextSection(
                            priority=2,
                            label="Active Censors",
                            content=censor_text,
                            token_estimate=self._estimate_tokens(censor_text),
                        )
                    )
            except Exception:
                logger.warning("Failed to load censors during context build")

        # 3. Frame
        if budget.frame > 0:
            frame_text = self._format_frame(frame)
            frame_text = self._truncate_to_budget(frame_text, budget.frame)
            sections.append(
                ContextSection(
                    priority=3,
                    label="Current Frame",
                    content=frame_text,
                    token_estimate=self._estimate_tokens(frame_text),
                )
            )

        # 4. Working memory (P1-7: no agent_id param)
        if budget.working_memory > 0:
            try:
                wm = await self._heart.get_working_memory(session_id, session=session)
                if wm is not None:
                    wm_text = self._format_working_memory(wm)
                    wm_text = self._truncate_to_budget(wm_text, budget.working_memory)
                    sections.append(
                        ContextSection(
                            priority=4,
                            label="Working Memory",
                            content=wm_text,
                            token_estimate=self._estimate_tokens(wm_text),
                        )
                    )
            except Exception:
                logger.warning("Failed to load working memory during context build")

        # 5. Decisions (P1-10: drop reasons from format)
        if budget.decisions > 0:
            try:
                decisions = await self._brain.query(input_text, limit=5, session=session)
                if decisions:
                    dec_text = self._format_decisions(decisions)
                    dec_text = self._truncate_to_budget(dec_text, budget.decisions)
                    sections.append(
                        ContextSection(
                            priority=5,
                            label="Related Decisions",
                            content=dec_text,
                            token_estimate=self._estimate_tokens(dec_text),
                        )
                    )
            except Exception:
                logger.warning("Brain.query failed during context build")

        # 6. Facts (P1-6: use type-specific search)
        if budget.facts > 0:
            try:
                facts = await self._heart.search_facts(input_text, limit=5, session=session)
                if facts:
                    facts = apply_frame_boost(facts, frame.frame_id, _active_censor_names)
                    facts_text = self._format_facts(facts)
                    facts_text = self._truncate_to_budget(facts_text, budget.facts)
                    sections.append(
                        ContextSection(
                            priority=6,
                            label="Relevant Facts",
                            content=facts_text,
                            token_estimate=self._estimate_tokens(facts_text),
                        )
                    )
            except Exception:
                logger.warning("Heart.search_facts failed during context build")

        # 7. Procedures
        if budget.procedures > 0:
            try:
                procedures = await self._heart.search_procedures(input_text, limit=5, session=session)
                if procedures:
                    procedures = apply_frame_boost(procedures, frame.frame_id, _active_censor_names)
                    proc_text = self._format_procedures(procedures)
                    proc_text = self._truncate_to_budget(proc_text, budget.procedures)
                    sections.append(
                        ContextSection(
                            priority=7,
                            label="Known Procedures",
                            content=proc_text,
                            token_estimate=self._estimate_tokens(proc_text),
                        )
                    )
            except Exception:
                logger.warning("Heart.search_procedures failed during context build")

        # 8. Episodes
        if budget.episodes > 0:
            try:
                episodes = await self._heart.search_episodes(input_text, limit=5, session=session)
                if episodes:
                    episodes = apply_frame_boost(episodes, frame.frame_id, _active_censor_names)
                    ep_text = self._format_episodes(episodes)
                    ep_text = self._truncate_to_budget(ep_text, budget.episodes)
                    sections.append(
                        ContextSection(
                            priority=8,
                            label="Past Episodes",
                            content=ep_text,
                            token_estimate=self._estimate_tokens(ep_text),
                        )
                    )
            except Exception:
                logger.warning("Heart.search_episodes failed during context build")

        # Assemble system prompt with markdown headers
        parts: list[str] = []
        for section in sorted(sections, key=lambda s: s.priority):
            parts.append(f"## {section.label}\n\n{section.content}")

        system_prompt = "\n\n".join(parts)
        return system_prompt, sections

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count: len(text) / CHARS_PER_TOKEN."""
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def _truncate_to_budget(self, text: str, token_budget: int) -> str:
        """Truncate text to fit within token budget."""
        max_chars = token_budget * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _format_decisions(self, decisions: list) -> str:
        """Format decision summaries for context.

        P1-10: No reasons field on DecisionSummary.
        Format: - [{outcome}] {description} (confidence: {confidence})
        """
        lines = []
        for d in decisions:
            outcome = getattr(d, "outcome", "pending") or "pending"
            desc = getattr(d, "description", "")
            conf = getattr(d, "confidence", 0.0)
            lines.append(f"- [{outcome}] {desc} (confidence: {conf:.2f})")
        return "\n".join(lines)

    def _format_facts(self, facts: list) -> str:
        """Format facts for context.

        P2-8: Use confidence + active (not confirmation_count + status).
        Format: - {content} [confidence: {confidence}, {active/inactive}]
        """
        lines = []
        for f in facts:
            content = getattr(f, "content", "")
            conf = getattr(f, "confidence", 1.0)
            active = getattr(f, "active", True)
            status = "active" if active else "inactive"
            lines.append(f"- {content} [confidence: {conf:.2f}, {status}]")
        return "\n".join(lines)

    def _format_procedures(self, procedures: list) -> str:
        """Format procedures for context.

        P2-8: Use name + domain + description (not core_patterns).
        ProcedureSummary has: name, domain, activation_count, effectiveness.
        Format: - **{name}** ({domain}): activated {count}x
        """
        lines = []
        for p in procedures:
            name = getattr(p, "name", "")
            domain = getattr(p, "domain", None) or "general"
            count = getattr(p, "activation_count", 0)
            eff = getattr(p, "effectiveness", None)
            eff_str = f", effectiveness: {eff:.0%}" if eff is not None else ""
            lines.append(f"- **{name}** ({domain}): activated {count}x{eff_str}")
        return "\n".join(lines)

    def _format_episodes(self, episodes: list) -> str:
        """Format episodes for context.

        Format: - [{outcome}] {summary} ({started_at date})
        """
        lines = []
        for e in episodes:
            outcome = getattr(e, "outcome", None) or "ongoing"
            summary = getattr(e, "summary", "")
            started = getattr(e, "started_at", None)
            date_str = started.strftime("%Y-%m-%d") if started else "unknown"
            lines.append(f"- [{outcome}] {summary} ({date_str})")
        return "\n".join(lines)

    def _format_censors(self, censors: list) -> str:
        """Format active censors.

        P1-4: Use action (not severity).
        Format: - **{ACTION}:** {trigger_pattern} -- {reason}
        """
        # Sort by action severity: block/absolute first, warn last
        action_order = {"absolute": 0, "block": 1, "warn": 2}
        sorted_censors = sorted(
            censors,
            key=lambda c: action_order.get(getattr(c, "action", "warn"), 3),
        )
        lines = []
        for c in sorted_censors:
            action = getattr(c, "action", "warn").upper()
            pattern = getattr(c, "trigger_pattern", "")
            reason = getattr(c, "reason", "")
            lines.append(f"- **{action}:** {pattern} -- {reason}")
        return "\n".join(lines)

    def _format_frame(self, frame: FrameSelection) -> str:
        """Format frame description and questions."""
        parts = [f"**{frame.frame_name}**: {frame.description or ''}"]
        if frame.questions_to_ask:
            parts.append("\nConsider asking:")
            for q in frame.questions_to_ask:
                parts.append(f"- {q}")
        return "\n".join(parts)

    def _format_working_memory(self, wm) -> str:
        """Format working memory state.

        Includes current_task, open_threads, and high-relevance items.
        """
        parts = []
        task = getattr(wm, "current_task", None)
        if task:
            parts.append(f"**Current task:** {task}")

        frame = getattr(wm, "current_frame", None)
        if frame:
            parts.append(f"**Frame:** {frame}")

        threads = getattr(wm, "open_threads", [])
        if threads:
            parts.append("\n**Open threads:**")
            for t in threads:
                desc = getattr(t, "description", "")
                priority = getattr(t, "priority", "medium")
                parts.append(f"- [{priority}] {desc}")

        items = getattr(wm, "items", [])
        # Only include high-relevance items (>= 0.7)
        high_rel = [i for i in items if getattr(i, "relevance", 0) >= 0.7]
        if high_rel:
            parts.append("\n**Loaded context:**")
            for item in high_rel:
                summary = getattr(item, "summary", "")
                rel = getattr(item, "relevance", 0)
                parts.append(f"- {summary} (relevance: {rel:.1f})")

        return "\n".join(parts) if parts else "No active working memory."

    async def refresh_needed(
        self,
        agent_id: str,
        session_id: str,
        new_input: str,
        current_frame: FrameSelection,
        session: AsyncSession | None = None,
    ) -> bool:
        """Check if context should be rebuilt.

        Returns True if:
        1. Working memory's current_frame differs from current_frame.frame_id
        2. No working memory exists for this session
        """
        try:
            wm = await self._heart.get_working_memory(session_id, session=session)
            if wm is None:
                return True
            return wm.current_frame != current_frame.frame_id
        except Exception:
            return True

    async def expand(
        self,
        memory_type: str,
        memory_id: str,
        session: AsyncSession | None = None,
    ) -> str:
        """Load full detail for a specific memory.

        memory_type: "decision", "fact", "episode", "procedure"
        Routes to Brain.get() or Heart managers accordingly.
        """
        uid = UUID(memory_id)

        if memory_type == "decision":
            detail = await self._brain.get(uid, session=session)
            if detail is None:
                return f"Decision {memory_id} not found."
            return (
                f"**{detail.description}**\n"
                f"Category: {detail.category} | Stakes: {detail.stakes} | "
                f"Confidence: {detail.confidence:.2f}\n"
                f"Context: {detail.context or 'None'}\n"
                f"Pattern: {detail.pattern or 'None'}"
            )

        if memory_type == "fact":
            detail = await self._heart.get_fact(uid, session=session)
            return (
                f"**{detail.content}**\n"
                f"Category: {detail.category or 'None'} | "
                f"Confidence: {detail.confidence:.2f} | "
                f"Source: {detail.source or 'unknown'}"
            )

        if memory_type == "episode":
            detail = await self._heart.get_episode(uid, session=session)
            return (
                f"**{detail.summary}**\n"
                f"Outcome: {detail.outcome or 'ongoing'} | "
                f"Started: {detail.started_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"Lessons: {', '.join(detail.lessons_learned) if detail.lessons_learned else 'None'}"
            )

        if memory_type == "procedure":
            detail = await self._heart.get_procedure(uid, session=session)
            return (
                f"**{detail.name}** ({detail.domain or 'general'})\n"
                f"{detail.description or ''}\n"
                f"Effectiveness: {detail.effectiveness or 'unknown'}"
            )

        return f"Unknown memory type: {memory_type}"
