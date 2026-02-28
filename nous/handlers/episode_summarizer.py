"""Episode Summarizer — generates structured summaries on session end.

Listens to: session_ended
Emits: episode_summarized

Uses a lightweight LLM call to summarize the conversation transcript.
Stores summary as JSONB on the episode record.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import httpx

from nous.config import Settings
from nous.events import Event, EventBus
from nous.handlers import build_anthropic_headers
from nous.heart.heart import Heart

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """You are summarizing a conversation episode for an AI agent's long-term memory.

Context:
- Agent: Nous (cognitive agent framework)
- This summary will be used for: semantic search recall, context assembly, calibration

Transcript:
{transcript}

{decision_context}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "title": "<5-10 word descriptive title focusing on WHAT WAS ACCOMPLISHED>",
  "summary": "<100-150 word prose summary emphasizing decisions made, problems solved, and outcomes>",
  "key_points": [
    "<lesson or reusable knowledge, not just event description>",
    "<pattern or insight that would help in similar future situations>"
  ],
  "outcome": "<resolved|partial|unresolved|informational>",
  "outcome_rationale": "<1 sentence explaining why this outcome classification>",
  "topics": ["<topic1>", "<topic2>"],
  "candidate_facts": [
    "<factual statement worth storing as long-term knowledge>"
  ]
}}

Outcome guidelines:
- resolved: The user's request was fully addressed, task completed, question answered
- partial: Work started but not finished, or only some requests addressed
- unresolved: Failed to complete the task, hit blockers
- informational: Casual chat, status check, no actionable work done

For key_points: Focus on WHAT WAS LEARNED, not what happened. Ask yourself:
"If this agent faces a similar situation, what from this episode would help?"

For candidate_facts: Extract concrete, reusable knowledge (tool configs, preferences,
architectural decisions, API behaviors) that should persist as standalone facts."""


class EpisodeSummarizer:
    """Generates episode summaries on session end.

    Listens to session_ended events. If the session had an active episode,
    fetches the transcript, calls LLM for summary, stores on episode record.
    """

    def __init__(
        self,
        heart: Heart,
        settings: Settings,
        bus: EventBus,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._heart = heart
        self._settings = settings
        self._bus = bus
        self._http = http_client
        bus.on("session_ended", self.handle)

    async def handle(self, event: Event) -> None:
        """Handle session_ended — summarize the episode if one exists."""
        episode_id = event.data.get("episode_id")
        if not episode_id:
            return

        try:
            # Fetch episode — skip if already summarized
            episode = await self._heart.get_episode(UUID(episode_id))
            if episode.structured_summary is not None:
                logger.debug("Episode %s already summarized, skipping", episode_id)
                return

            # Get transcript from event data
            transcript = event.data.get("transcript", "")
            if not transcript or len(transcript) < 50:
                logger.debug("Episode %s too short for summary, skipping", episode_id)
                return

            # Call LLM for summary
            summary = await self._generate_summary(transcript)
            if not summary:
                return

            # Store summary on episode
            await self._heart.update_episode_summary(UUID(episode_id), summary)

            # Emit for downstream handlers (fact extraction)
            await self._bus.emit(Event(
                type="episode_summarized",
                agent_id=event.agent_id,
                session_id=event.session_id,
                data={
                    "episode_id": episode_id,
                    "summary": summary,
                    "candidate_facts": summary.get("candidate_facts", []),
                },
            ))

            logger.info("Episode %s summarized: %s", episode_id, summary.get("title", "?"))

        except Exception:
            logger.exception("Failed to summarize episode %s", episode_id)

    async def _generate_summary(self, transcript: str) -> dict[str, Any] | None:
        """Call LLM to generate structured summary."""
        if not self._http:
            logger.warning("No HTTP client for episode summarizer")
            return None

        # Truncate transcript to ~8000 chars
        if len(transcript) > 8000:
            half = 3800
            transcript = transcript[:half] + "\n\n[... middle truncated ...]\n\n" + transcript[-half:]

        prompt = _SUMMARY_PROMPT.format(transcript=transcript, decision_context="")
        headers = build_anthropic_headers(self._settings)

        try:
            response = await self._http.post(
                f"{self._settings.api_base_url}/v1/messages",
                json={
                    "model": self._settings.background_model,
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers=headers,
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning("Summary LLM call failed: %d", response.status_code)
                return None

            data = response.json()
            text = data.get("content", [{}])[0].get("text", "")

            return json.loads(text)

        except (json.JSONDecodeError, httpx.TimeoutException) as e:
            logger.warning("Summary generation failed: %s", e)
            return None
