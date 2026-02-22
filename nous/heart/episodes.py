"""Episode management — what happened.

Manages episodic memory: starting, ending, linking, searching episodes.
All methods follow Brain's session injection pattern (P1-1).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from nous.brain.embeddings import EmbeddingProvider
from nous.heart.schemas import EpisodeDetail, EpisodeInput, EpisodeSummary
from nous.heart.search import hybrid_search
from nous.storage.database import Database
from nous.storage.models import Episode, EpisodeDecision, EpisodeProcedure, Event

logger = logging.getLogger(__name__)


class EpisodeManager:
    """Manages episodic memory — what happened."""

    def __init__(
        self,
        db: Database,
        embeddings: EmbeddingProvider | None,
        agent_id: str,
    ) -> None:
        self.db = db
        self.embeddings = embeddings
        self.agent_id = agent_id

    # ------------------------------------------------------------------
    # Event helper
    # ------------------------------------------------------------------

    async def _emit_event(
        self, session: AsyncSession, event_type: str, data: dict
    ) -> None:
        """Insert event in same session (P2-1)."""
        event = Event(
            agent_id=self.agent_id,
            event_type=event_type,
            data=data,
        )
        session.add(event)

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    async def start(
        self, input: EpisodeInput, session: AsyncSession | None = None
    ) -> EpisodeDetail:
        """Start a new episode."""
        if session is None:
            async with self.db.session() as session:
                result = await self._start(input, session)
                await session.commit()
                return result
        return await self._start(input, session)

    async def _start(
        self, input: EpisodeInput, session: AsyncSession
    ) -> EpisodeDetail:
        # Generate embedding from title + summary
        embedding = None
        if self.embeddings:
            embed_text = f"{input.title or ''} {input.summary}".strip()
            try:
                embedding = await self.embeddings.embed(embed_text)
            except Exception:
                logger.warning("Embedding generation failed for episode start")

        episode = Episode(
            agent_id=self.agent_id,
            title=input.title,
            summary=input.summary,
            detail=input.detail,
            frame_used=input.frame_used,
            trigger=input.trigger,
            participants=input.participants or None,
            tags=input.tags or None,
            embedding=embedding,
        )
        session.add(episode)
        await session.flush()

        await self._emit_event(
            session, "episode_started", {"episode_id": str(episode.id)}
        )

        # Re-fetch with eager loading to avoid lazy-load MissingGreenlet
        episode = await self._get_episode_orm(episode.id, session)
        return self._to_detail(episode)

    # ------------------------------------------------------------------
    # end()
    # ------------------------------------------------------------------

    async def end(
        self,
        episode_id: UUID,
        outcome: str,
        lessons_learned: list[str] | None = None,
        surprise_level: float | None = None,
        session: AsyncSession | None = None,
    ) -> EpisodeDetail:
        """Close an episode with outcome and lessons."""
        if session is None:
            async with self.db.session() as session:
                result = await self._end(
                    episode_id, outcome, lessons_learned, surprise_level, session
                )
                await session.commit()
                return result
        return await self._end(
            episode_id, outcome, lessons_learned, surprise_level, session
        )

    async def _end(
        self,
        episode_id: UUID,
        outcome: str,
        lessons_learned: list[str] | None,
        surprise_level: float | None,
        session: AsyncSession,
    ) -> EpisodeDetail:
        episode = await self._get_episode_orm(episode_id, session)
        if episode is None:
            raise ValueError(f"Episode {episode_id} not found")

        now = datetime.now(timezone.utc)
        episode.ended_at = now
        episode.duration_seconds = int((now - episode.started_at).total_seconds())
        episode.outcome = outcome
        episode.lessons_learned = lessons_learned
        episode.surprise_level = surprise_level

        # P3-7: Regenerate embedding incorporating outcome + lessons
        if self.embeddings:
            embed_text = (
                f"{episode.title or ''} {episode.summary} "
                f"{outcome} {' '.join(lessons_learned or [])}"
            ).strip()
            try:
                episode.embedding = await self.embeddings.embed(embed_text)
            except Exception:
                logger.warning("Embedding regeneration failed for episode end")

        await session.flush()

        await self._emit_event(
            session,
            "episode_completed",
            {
                "episode_id": str(episode_id),
                "outcome": outcome,
                "duration": episode.duration_seconds,
            },
        )

        # Re-fetch with eager loading to avoid lazy-load MissingGreenlet
        episode = await self._get_episode_orm(episode_id, session)
        return self._to_detail(episode)

    # ------------------------------------------------------------------
    # link_decision()
    # ------------------------------------------------------------------

    async def link_decision(
        self,
        episode_id: UUID,
        decision_id: UUID,
        session: AsyncSession | None = None,
    ) -> None:
        """Insert into heart.episode_decisions."""
        if session is None:
            async with self.db.session() as session:
                await self._link_decision(episode_id, decision_id, session)
                await session.commit()
                return
        await self._link_decision(episode_id, decision_id, session)

    async def _link_decision(
        self,
        episode_id: UUID,
        decision_id: UUID,
        session: AsyncSession,
    ) -> None:
        link = EpisodeDecision(episode_id=episode_id, decision_id=decision_id)
        session.add(link)
        await session.flush()

    # ------------------------------------------------------------------
    # link_procedure()
    # ------------------------------------------------------------------

    async def link_procedure(
        self,
        episode_id: UUID,
        procedure_id: UUID,
        effectiveness: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Insert into heart.episode_procedures with optional effectiveness."""
        if session is None:
            async with self.db.session() as session:
                await self._link_procedure(
                    episode_id, procedure_id, effectiveness, session
                )
                await session.commit()
                return
        await self._link_procedure(episode_id, procedure_id, effectiveness, session)

    async def _link_procedure(
        self,
        episode_id: UUID,
        procedure_id: UUID,
        effectiveness: str | None,
        session: AsyncSession,
    ) -> None:
        link = EpisodeProcedure(
            episode_id=episode_id,
            procedure_id=procedure_id,
            effectiveness=effectiveness,
        )
        session.add(link)
        await session.flush()

    # ------------------------------------------------------------------
    # get()
    # ------------------------------------------------------------------

    async def get(
        self, episode_id: UUID, session: AsyncSession | None = None
    ) -> EpisodeDetail | None:
        """Fetch episode with linked decision_ids."""
        if session is None:
            async with self.db.session() as session:
                return await self._get(episode_id, session)
        return await self._get(episode_id, session)

    async def _get(
        self, episode_id: UUID, session: AsyncSession
    ) -> EpisodeDetail | None:
        episode = await self._get_episode_orm(episode_id, session)
        if episode is None:
            return None
        return self._to_detail(episode)

    # ------------------------------------------------------------------
    # list_recent()
    # ------------------------------------------------------------------

    async def list_recent(
        self,
        limit: int = 10,
        outcome: str | None = None,
        session: AsyncSession | None = None,
    ) -> list[EpisodeSummary]:
        """List recent episodes ordered by started_at DESC."""
        if session is None:
            async with self.db.session() as session:
                return await self._list_recent(limit, outcome, session)
        return await self._list_recent(limit, outcome, session)

    async def _list_recent(
        self,
        limit: int,
        outcome: str | None,
        session: AsyncSession,
    ) -> list[EpisodeSummary]:
        stmt = (
            select(Episode)
            .where(Episode.agent_id == self.agent_id)
            .where(Episode.active == True)  # noqa: E712
            .order_by(Episode.started_at.desc())
            .limit(limit)
        )
        if outcome is not None:
            stmt = stmt.where(Episode.outcome == outcome)

        result = await session.execute(stmt)
        episodes = result.scalars().all()

        return [
            EpisodeSummary(
                id=e.id,
                title=e.title,
                summary=e.summary,
                outcome=e.outcome,
                started_at=e.started_at,
                tags=e.tags or [],
            )
            for e in episodes
        ]

    # ------------------------------------------------------------------
    # search()
    # ------------------------------------------------------------------

    async def search(
        self, query: str, limit: int = 10, session: AsyncSession | None = None
    ) -> list[EpisodeSummary]:
        """Hybrid search over episodes using search.py helper."""
        if session is None:
            async with self.db.session() as session:
                return await self._search(query, limit, session)
        return await self._search(query, limit, session)

    async def _search(
        self, query: str, limit: int, session: AsyncSession
    ) -> list[EpisodeSummary]:
        # Generate query embedding
        embedding = None
        if self.embeddings:
            try:
                embedding = await self.embeddings.embed(query)
            except Exception:
                logger.warning("Embedding generation failed for episode search")

        results = await hybrid_search(
            session=session,
            table="heart.episodes",
            embedding=embedding,
            query_text=query,
            agent_id=self.agent_id,
            limit=limit,
        )

        if not results:
            return []

        # Fetch full episodes by IDs
        ids = [r[0] for r in results]
        scores = {r[0]: r[1] for r in results}

        ep_result = await session.execute(
            select(Episode).where(Episode.id.in_(ids))
        )
        episodes = {e.id: e for e in ep_result.scalars().all()}

        # Preserve search order
        return [
            EpisodeSummary(
                id=e.id,
                title=e.title,
                summary=e.summary,
                outcome=e.outcome,
                started_at=e.started_at,
                tags=e.tags or [],
                score=scores.get(e.id),
            )
            for eid in ids
            if (e := episodes.get(eid)) is not None
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_episode_orm(
        self, episode_id: UUID, session: AsyncSession
    ) -> Episode | None:
        """Fetch Episode ORM with eager-loaded relationships, scoped by agent_id."""
        result = await session.execute(
            select(Episode)
            .options(selectinload(Episode.episode_decisions))
            .where(Episode.id == episode_id)
            .where(Episode.agent_id == self.agent_id)
        )
        return result.scalars().first()

    def _to_detail(self, episode: Episode) -> EpisodeDetail:
        """Convert ORM Episode to EpisodeDetail DTO."""
        decision_ids = [
            ed.decision_id for ed in (episode.episode_decisions or [])
        ]
        return EpisodeDetail(
            id=episode.id,
            agent_id=episode.agent_id,
            title=episode.title,
            summary=episode.summary,
            detail=episode.detail,
            started_at=episode.started_at,
            ended_at=episode.ended_at,
            duration_seconds=episode.duration_seconds,
            frame_used=episode.frame_used,
            trigger=episode.trigger,
            participants=episode.participants or [],
            outcome=episode.outcome,
            surprise_level=episode.surprise_level,
            lessons_learned=episode.lessons_learned or [],
            tags=episode.tags or [],
            decision_ids=decision_ids,
            created_at=episode.created_at,
        )
