"""Tests for schedule management (011.1)."""

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from nous.heart.schedules import ScheduleManager
from nous.storage.models import Schedule


@pytest_asyncio.fixture
async def session(db):
    async with db.engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        yield session
        await session.close()
        await trans.rollback()


@pytest_asyncio.fixture
async def schedule_mgr(db):
    return ScheduleManager(db, "test-agent")


class TestScheduleManager:

    async def test_create_once_schedule(self, schedule_mgr: ScheduleManager):
        fire_at = datetime.now(UTC) + timedelta(hours=2)
        schedule = await schedule_mgr.create(
            task="Remind Tim",
            schedule_type="once",
            fire_at=fire_at,
        )
        assert schedule.schedule_type == "once"
        assert schedule.next_fire_at == fire_at
        assert schedule.active is True

    async def test_create_recurring_with_interval(self, schedule_mgr: ScheduleManager):
        schedule = await schedule_mgr.create(
            task="Check snow",
            schedule_type="recurring",
            interval_seconds=21600,
        )
        assert schedule.schedule_type == "recurring"
        assert schedule.interval_seconds == 21600
        assert schedule.next_fire_at is not None

    async def test_create_recurring_with_cron(self, schedule_mgr: ScheduleManager):
        schedule = await schedule_mgr.create(
            task="Daily check",
            schedule_type="recurring",
            cron_expr="0 8 * * *",
        )
        assert schedule.cron_expr == "0 8 * * *"
        assert schedule.next_fire_at is not None

    async def test_create_recurring_without_timing_raises(self, schedule_mgr: ScheduleManager):
        with pytest.raises(ValueError, match="interval_seconds or cron_expr"):
            await schedule_mgr.create(
                task="Bad schedule",
                schedule_type="recurring",
            )

    async def test_get_due_schedules(self, schedule_mgr: ScheduleManager):
        # Create schedule due in the past
        past = datetime.now(UTC) - timedelta(minutes=5)
        await schedule_mgr.create(
            task="Overdue task",
            schedule_type="once",
            fire_at=past,
        )
        # Create schedule due in the future
        future = datetime.now(UTC) + timedelta(hours=2)
        await schedule_mgr.create(
            task="Future task",
            schedule_type="once",
            fire_at=future,
        )

        due = await schedule_mgr.get_due(datetime.now(UTC))
        assert len(due) == 1
        assert due[0].task == "Overdue task"

    async def test_advance_recurring_schedule(self, schedule_mgr: ScheduleManager):
        schedule = await schedule_mgr.create(
            task="Recurring",
            schedule_type="recurring",
            interval_seconds=3600,
        )
        now = datetime.now(UTC)
        await schedule_mgr.advance(schedule.id, now)

        updated = await schedule_mgr.get(schedule.id)
        assert updated.fire_count == 1
        assert updated.last_fired_at is not None
        # next_fire_at should be ~1 hour from now
        assert updated.next_fire_at > now

    async def test_advance_deactivates_at_max_fires(self, schedule_mgr: ScheduleManager):
        schedule = await schedule_mgr.create(
            task="Limited",
            schedule_type="recurring",
            interval_seconds=3600,
            max_fires=1,
        )
        await schedule_mgr.advance(schedule.id, datetime.now(UTC))

        updated = await schedule_mgr.get(schedule.id)
        assert updated.fire_count == 1
        assert updated.active is False

    async def test_deactivate_once_schedule(self, schedule_mgr: ScheduleManager):
        fire_at = datetime.now(UTC) + timedelta(hours=1)
        schedule = await schedule_mgr.create(
            task="One shot",
            schedule_type="once",
            fire_at=fire_at,
        )
        await schedule_mgr.deactivate(schedule.id)

        updated = await schedule_mgr.get(schedule.id)
        assert updated.active is False

    async def test_list_active_schedules(self, schedule_mgr: ScheduleManager):
        await schedule_mgr.create(
            task="Active",
            schedule_type="recurring",
            interval_seconds=3600,
        )
        s2 = await schedule_mgr.create(
            task="Inactive",
            schedule_type="once",
            fire_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await schedule_mgr.deactivate(s2.id)

        active = await schedule_mgr.list(active_only=True)
        assert len(active) == 1
        assert active[0].task == "Active"

    async def test_list_all_schedules(self, schedule_mgr: ScheduleManager):
        await schedule_mgr.create(
            task="A", schedule_type="recurring", interval_seconds=3600,
        )
        s2 = await schedule_mgr.create(
            task="B", schedule_type="once",
            fire_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await schedule_mgr.deactivate(s2.id)

        all_sched = await schedule_mgr.list(active_only=False)
        assert len(all_sched) == 2

    async def test_get_schedule(self, schedule_mgr: ScheduleManager):
        schedule = await schedule_mgr.create(
            task="Get me",
            schedule_type="recurring",
            interval_seconds=3600,
        )
        fetched = await schedule_mgr.get(schedule.id)
        assert fetched is not None
        assert fetched.task == "Get me"

    async def test_get_nonexistent_schedule(self, schedule_mgr: ScheduleManager):
        import uuid
        result = await schedule_mgr.get(uuid.uuid4())
        assert result is None

    async def test_advance_cron_schedule(self, schedule_mgr: ScheduleManager):
        schedule = await schedule_mgr.create(
            task="Cron recurring",
            schedule_type="recurring",
            cron_expr="0 */6 * * *",  # Every 6 hours
        )
        now = datetime.now(UTC)
        await schedule_mgr.advance(schedule.id, now)

        updated = await schedule_mgr.get(schedule.id)
        assert updated.fire_count == 1
        assert updated.next_fire_at > now
