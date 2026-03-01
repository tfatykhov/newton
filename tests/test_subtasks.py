"""Tests for subtask queue and scheduling (011.1)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from nous.storage.models import Schedule, Subtask


@pytest_asyncio.fixture
async def session(db):
    """Function-scoped session with transaction rollback."""
    async with db.engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        yield session
        await session.close()
        await trans.rollback()


class TestSubtaskModel:
    """ORM model tests for heart.subtasks."""

    async def test_create_subtask(self, session: AsyncSession):
        subtask = Subtask(
            agent_id="test-agent",
            task="Research snow conditions",
            priority=100,
            timeout_seconds=120,
        )
        session.add(subtask)
        await session.flush()

        assert subtask.id is not None
        assert subtask.status == "pending"
        assert subtask.notify is True
        assert subtask.created_at is not None

    async def test_subtask_with_parent_session(self, session: AsyncSession):
        subtask = Subtask(
            agent_id="test-agent",
            parent_session_id="session-abc123",
            task="Check lift ticket prices",
            priority=50,
        )
        session.add(subtask)
        await session.flush()

        assert subtask.parent_session_id == "session-abc123"
        assert subtask.priority == 50


class TestScheduleModel:
    """ORM model tests for heart.schedules."""

    async def test_create_once_schedule(self, session: AsyncSession):
        fire_time = datetime.now(UTC) + timedelta(hours=2)
        schedule = Schedule(
            agent_id="test-agent",
            task="Remind Tim about hotel",
            schedule_type="once",
            fire_at=fire_time,
            next_fire_at=fire_time,
        )
        session.add(schedule)
        await session.flush()

        assert schedule.id is not None
        assert schedule.schedule_type == "once"
        assert schedule.active is True
        assert schedule.fire_count == 0

    async def test_create_recurring_schedule(self, session: AsyncSession):
        schedule = Schedule(
            agent_id="test-agent",
            task="Check snow conditions",
            schedule_type="recurring",
            interval_seconds=21600,  # 6 hours
            next_fire_at=datetime.now(UTC) + timedelta(hours=6),
        )
        session.add(schedule)
        await session.flush()

        assert schedule.schedule_type == "recurring"
        assert schedule.interval_seconds == 21600
