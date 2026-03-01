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


# ---------------------------------------------------------------------------
# SubtaskManager tests
# ---------------------------------------------------------------------------

from nous.heart.subtasks import SubtaskManager


@pytest_asyncio.fixture
async def subtask_mgr(db):
    return SubtaskManager(db, "test-agent")


class TestSubtaskManager:
    """SubtaskManager CRUD and queue tests."""

    async def test_create_subtask(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(
            task="Research snow conditions",
            parent_session_id="session-123",
            priority="normal",
            timeout=120,
            notify=True,
        )
        assert subtask.id is not None
        assert subtask.status == "pending"
        assert subtask.priority == 100
        assert subtask.task == "Research snow conditions"

    async def test_create_urgent_priority(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(
            task="Urgent task", priority="urgent",
        )
        assert subtask.priority == 50

    async def test_create_low_priority(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(
            task="Low priority task", priority="low",
        )
        assert subtask.priority == 200

    async def test_create_rejects_when_too_many_pending(self, subtask_mgr: SubtaskManager):
        for i in range(5):
            await subtask_mgr.create(task=f"Task {i}")
        with pytest.raises(ValueError, match="pending subtask limit"):
            await subtask_mgr.create(task="One too many")

    async def test_dequeue_returns_highest_priority(self, subtask_mgr: SubtaskManager):
        await subtask_mgr.create(task="Normal task", priority="normal")
        await subtask_mgr.create(task="Urgent task", priority="urgent")

        subtask = await subtask_mgr.dequeue("worker-0")
        assert subtask is not None
        assert subtask.task == "Urgent task"
        assert subtask.status == "running"
        assert subtask.worker_id == "worker-0"

    async def test_dequeue_returns_none_when_empty(self, subtask_mgr: SubtaskManager):
        result = await subtask_mgr.dequeue("worker-0")
        assert result is None

    async def test_complete_subtask(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(task="Test task")
        await subtask_mgr.dequeue("worker-0")

        await subtask_mgr.complete(subtask.id, "Task result here")
        updated = await subtask_mgr.get(subtask.id)
        assert updated.status == "completed"
        assert updated.result == "Task result here"
        assert updated.completed_at is not None

    async def test_fail_subtask(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(task="Failing task")
        await subtask_mgr.dequeue("worker-0")

        await subtask_mgr.fail(subtask.id, "Timeout exceeded")
        updated = await subtask_mgr.get(subtask.id)
        assert updated.status == "failed"
        assert updated.error == "Timeout exceeded"

    async def test_cancel_pending_subtask(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(task="Cancel me")
        cancelled = await subtask_mgr.cancel(subtask.id)
        assert cancelled is True
        updated = await subtask_mgr.get(subtask.id)
        assert updated.status == "cancelled"

    async def test_cancel_running_subtask_fails(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(task="Running task")
        await subtask_mgr.dequeue("worker-0")
        cancelled = await subtask_mgr.cancel(subtask.id)
        assert cancelled is False

    async def test_list_subtasks(self, subtask_mgr: SubtaskManager):
        await subtask_mgr.create(task="Task A")
        await subtask_mgr.create(task="Task B")
        results = await subtask_mgr.list(limit=10)
        assert len(results) == 2

    async def test_list_by_status(self, subtask_mgr: SubtaskManager):
        await subtask_mgr.create(task="Task A")
        await subtask_mgr.create(task="Task B")
        await subtask_mgr.dequeue("worker-0")  # dequeues Task A (first created)

        pending = await subtask_mgr.list(status="pending", limit=10)
        assert len(pending) == 1
        running = await subtask_mgr.list(status="running", limit=10)
        assert len(running) == 1

    async def test_reclaim_stale(self, subtask_mgr: SubtaskManager):
        subtask = await subtask_mgr.create(task="Stale task", timeout=1)
        await subtask_mgr.dequeue("worker-0")
        # Manually backdate started_at to simulate stale
        async with subtask_mgr._db.session() as sess:
            from sqlalchemy import update
            from nous.storage.models import Subtask as SubtaskModel
            await sess.execute(
                update(SubtaskModel)
                .where(SubtaskModel.id == subtask.id)
                .values(started_at=datetime.now(UTC) - timedelta(seconds=300))
            )
            await sess.commit()

        reclaimed = await subtask_mgr.reclaim_stale()
        assert reclaimed >= 1
        updated = await subtask_mgr.get(subtask.id)
        assert updated.status == "pending"

    async def test_count_by_status(self, subtask_mgr: SubtaskManager):
        await subtask_mgr.create(task="A")
        await subtask_mgr.create(task="B")
        await subtask_mgr.dequeue("w-0")

        counts = await subtask_mgr.count_by_status()
        assert counts["pending"] == 1
        assert counts["running"] == 1
