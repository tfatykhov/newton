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

_SUMMARY_PROMPT = """Given the following conversation transcript, generate a structured summary.

Transcript:
{transcript}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "title": "<5-10 word descriptive title>",
  "summary": "<100-150 word prose summary of what happened>",
  "key_points": ["<point 1>", "<point 2>", "<point 3>"],
  "outcome": "<resolved|partial|unresolved|informational>",
  "topics": ["<topic1>", "<topic2>"]
}}"""


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

        prompt = _SUMMARY_PROMPT.format(transcript=transcript)
        headers = build_anthropic_headers(self._settings)

        try:
            response = await self._http.post(
                f"{self._settings.api_base_url}/v1/messages",
                json={
                    "model": self._settings.background_model,
                    "max_tokens": 500,
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
