"""Tests for subtask queue, scheduling, and worker pool (011.1)."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from nous.config import Settings
from nous.handlers.subtask_worker import SubtaskWorkerPool
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


# ---------------------------------------------------------------------------
# Heart integration tests
# ---------------------------------------------------------------------------

from nous.heart.heart import Heart


class TestHeartIntegration:
    """Verify Heart exposes subtask and schedule managers."""

    async def test_heart_has_subtask_manager(self, db, settings):
        heart = Heart(db, settings)
        assert heart.subtasks is not None
        assert hasattr(heart.subtasks, "create")
        assert hasattr(heart.subtasks, "dequeue")
        assert hasattr(heart.subtasks, "complete")
        await heart.close()

    async def test_heart_has_schedule_manager(self, db, settings):
        heart = Heart(db, settings)
        assert heart.schedules is not None
        assert hasattr(heart.schedules, "create")
        assert hasattr(heart.schedules, "get_due")
        assert hasattr(heart.schedules, "advance")
        await heart.close()


# ---------------------------------------------------------------------------
# SubtaskWorkerPool tests
# ---------------------------------------------------------------------------


class TestSubtaskWorkerPool:
    """Worker pool tests using mocked runner and bus."""

    @pytest.fixture
    def mock_runner(self):
        runner = AsyncMock()
        runner.run_turn = AsyncMock(return_value=(
            "Task completed successfully",
            MagicMock(),  # TurnContext
            {"input_tokens": 100, "output_tokens": 50},
        ))
        return runner

    @pytest.fixture
    def mock_bus(self):
        bus = AsyncMock()
        bus.emit = AsyncMock()
        return bus

    @pytest.fixture
    def worker_settings(self):
        return Settings(
            subtask_workers=1,
            subtask_poll_interval=0.1,
            subtask_default_timeout=120,
            subtask_max_concurrent=3,
            telegram_bot_token=None,
            telegram_chat_id=None,
        )

    @pytest_asyncio.fixture
    async def worker_heart(self, db, worker_settings):
        heart = Heart(db, worker_settings)
        yield heart
        await heart.close()

    async def test_execute_subtask_success(
        self, mock_runner, worker_heart, worker_settings, mock_bus
    ):
        """Executing a subtask calls runner.run_turn and marks complete."""
        pool = SubtaskWorkerPool(
            runner=mock_runner,
            heart=worker_heart,
            settings=worker_settings,
            bus=mock_bus,
        )

        subtask = await worker_heart.subtasks.create(task="Test background work")
        # Dequeue to set status to running (as the worker loop would)
        dequeued = await worker_heart.subtasks.dequeue("test-worker")
        assert dequeued is not None

        await pool._execute_subtask(dequeued)

        # Runner was called with correct session_id
        mock_runner.run_turn.assert_awaited_once()
        call_kwargs = mock_runner.run_turn.call_args
        assert call_kwargs.kwargs["session_id"] == f"subtask-{subtask.id.hex[:8]}"
        assert call_kwargs.kwargs["user_message"] == "Test background work"

        # Subtask marked complete
        updated = await worker_heart.subtasks.get(subtask.id)
        assert updated.status == "completed"
        assert updated.result == "Task completed successfully"

        # Event emitted
        mock_bus.emit.assert_awaited()
        emitted_event = mock_bus.emit.call_args[0][0]
        assert emitted_event.type == "subtask_completed"

    async def test_execute_subtask_failure(
        self, mock_runner, worker_heart, worker_settings, mock_bus
    ):
        """A failing runner marks the subtask as failed and emits error event."""
        mock_runner.run_turn = AsyncMock(side_effect=RuntimeError("LLM API down"))
        pool = SubtaskWorkerPool(
            runner=mock_runner,
            heart=worker_heart,
            settings=worker_settings,
            bus=mock_bus,
        )

        subtask = await worker_heart.subtasks.create(task="Doomed task")
        dequeued = await worker_heart.subtasks.dequeue("test-worker")

        await pool._execute_subtask(dequeued)

        updated = await worker_heart.subtasks.get(subtask.id)
        assert updated.status == "failed"
        assert "RuntimeError" in updated.error

        mock_bus.emit.assert_awaited()
        emitted_event = mock_bus.emit.call_args[0][0]
        assert emitted_event.type == "subtask_failed"

    async def test_process_subtask_timeout(
        self, worker_heart, worker_settings, mock_bus
    ):
        """A subtask exceeding its timeout is marked failed."""
        # Runner that takes too long
        async def slow_turn(**kwargs):
            await asyncio.sleep(10)
            return ("done", MagicMock(), {})

        slow_runner = AsyncMock()
        slow_runner.run_turn = slow_turn

        pool = SubtaskWorkerPool(
            runner=slow_runner,
            heart=worker_heart,
            settings=worker_settings,
            bus=mock_bus,
        )

        subtask = await worker_heart.subtasks.create(task="Slow task", timeout=1)
        dequeued = await worker_heart.subtasks.dequeue("test-worker")

        await pool._process_subtask(dequeued)

        updated = await worker_heart.subtasks.get(subtask.id)
        assert updated.status == "failed"
        assert "timed out" in updated.error

    async def test_execute_subtask_no_bus(
        self, mock_runner, worker_heart, worker_settings
    ):
        """Worker pool works without event bus (bus=None)."""
        pool = SubtaskWorkerPool(
            runner=mock_runner,
            heart=worker_heart,
            settings=worker_settings,
            bus=None,
        )

        subtask = await worker_heart.subtasks.create(task="No bus task")
        dequeued = await worker_heart.subtasks.dequeue("test-worker")

        await pool._execute_subtask(dequeued)

        updated = await worker_heart.subtasks.get(subtask.id)
        assert updated.status == "completed"

    async def test_start_reclaims_stale(
        self, mock_runner, worker_heart, worker_settings
    ):
        """start() calls reclaim_stale on startup."""
        pool = SubtaskWorkerPool(
            runner=mock_runner,
            heart=worker_heart,
            settings=worker_settings,
        )

        # Spy on reclaim_stale
        original_reclaim = worker_heart.subtasks.reclaim_stale
        reclaim_called = False

        async def spy_reclaim():
            nonlocal reclaim_called
            reclaim_called = True
            return await original_reclaim()

        worker_heart.subtasks.reclaim_stale = spy_reclaim

        await pool.start()
        assert reclaim_called
        await pool.stop()

    async def test_start_and_stop_workers(
        self, mock_runner, worker_heart, worker_settings
    ):
        """start() spawns workers, stop() cancels them."""
        pool = SubtaskWorkerPool(
            runner=mock_runner,
            heart=worker_heart,
            settings=worker_settings,
        )

        await pool.start()
        assert len(pool._workers) == 1  # subtask_workers=1 in fixture

        await pool.stop()
        assert len(pool._workers) == 0

    async def test_telegram_notification_on_success(
        self, mock_runner, worker_heart, worker_settings
    ):
        """Telegram notification sent on successful completion when configured."""
        worker_settings.telegram_bot_token = "test-token"
        worker_settings.telegram_chat_id = "12345"

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))

        pool = SubtaskWorkerPool(
            runner=mock_runner,
            heart=worker_heart,
            settings=worker_settings,
            http_client=mock_http,
        )

        subtask = await worker_heart.subtasks.create(task="Notify me", notify=True)
        dequeued = await worker_heart.subtasks.dequeue("test-worker")

        await pool._execute_subtask(dequeued)

        mock_http.post.assert_awaited()
        call_args = mock_http.post.call_args
        assert "sendMessage" in call_args[0][0]
        body = call_args.kwargs["json"]
        assert body["chat_id"] == "12345"
        assert "completed" in body["text"].lower()

    async def test_no_telegram_when_notify_false(
        self, mock_runner, worker_heart, worker_settings
    ):
        """No Telegram notification when subtask.notify is False."""
        worker_settings.telegram_bot_token = "test-token"
        worker_settings.telegram_chat_id = "12345"

        mock_http = AsyncMock()
        pool = SubtaskWorkerPool(
            runner=mock_runner,
            heart=worker_heart,
            settings=worker_settings,
            http_client=mock_http,
        )

        subtask = await worker_heart.subtasks.create(task="Silent task", notify=False)
        dequeued = await worker_heart.subtasks.dequeue("test-worker")

        await pool._execute_subtask(dequeued)

        mock_http.post.assert_not_awaited()
