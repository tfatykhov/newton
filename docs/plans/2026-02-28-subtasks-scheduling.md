# 011.1 Subtasks & Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give Nous background subtask execution and scheduled/recurring task capabilities.

**Architecture:** Postgres-backed task queue with async worker pool. Subtasks execute as independent agent turns via existing AgentRunner. Scheduler is a background loop that enqueues due tasks. Both share the heart schema and integrate with the event bus for notifications.

**Tech Stack:** Python 3.12+, SQLAlchemy async ORM, asyncpg, croniter (new dep), python-dateutil (existing), httpx for Telegram notifications.

---

## Task 1: Add croniter dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add croniter to dependencies**

```bash
uv add croniter
```

**Step 2: Verify installation**

```bash
uv run python -c "import croniter; print(croniter.__version__)"
```
Expected: version string printed, no errors.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add croniter dependency for schedule cron support"
```

---

## Task 2: Database migration — subtasks and schedules tables

**Files:**
- Create: `sql/migrations/010_subtasks.sql`

**Step 1: Write the migration**

```sql
-- 011.1: Subtasks & Scheduling (F009)

-- Subtask queue
CREATE TABLE IF NOT EXISTS heart.subtasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    parent_session_id VARCHAR,
    task TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    worker_id VARCHAR(100),
    timeout_seconds INTEGER NOT NULL DEFAULT 120,
    notify BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT chk_subtask_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_subtask_priority CHECK (priority > 0)
);

CREATE INDEX IF NOT EXISTS idx_subtasks_pending
    ON heart.subtasks (priority, created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_subtasks_agent
    ON heart.subtasks (agent_id, created_at DESC);

-- Schedule table
CREATE TABLE IF NOT EXISTS heart.schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    task TEXT NOT NULL,
    schedule_type VARCHAR(20) NOT NULL,
    fire_at TIMESTAMPTZ,
    interval_seconds INTEGER,
    cron_expr VARCHAR(200),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    last_fired_at TIMESTAMPTZ,
    next_fire_at TIMESTAMPTZ,
    fire_count INTEGER NOT NULL DEFAULT 0,
    max_fires INTEGER,
    notify BOOLEAN NOT NULL DEFAULT TRUE,
    timeout_seconds INTEGER NOT NULL DEFAULT 120,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by_session VARCHAR,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT chk_schedule_type CHECK (schedule_type IN ('once', 'recurring')),
    CONSTRAINT chk_schedule_has_timing CHECK (
        (schedule_type = 'once' AND fire_at IS NOT NULL) OR
        (schedule_type = 'recurring' AND (interval_seconds IS NOT NULL OR cron_expr IS NOT NULL))
    )
);

CREATE INDEX IF NOT EXISTS idx_schedules_next
    ON heart.schedules (next_fire_at) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_schedules_agent
    ON heart.schedules (agent_id) WHERE active = TRUE;
```

**Step 2: Verify migration runs**

```bash
uv run python -c "
import asyncio
from nous.config import Settings
from nous.storage.database import Database
from nous.main import run_migrations
async def test():
    s = Settings()
    db = Database(s)
    await db.connect()
    await run_migrations(db.engine)
    await db.disconnect()
asyncio.run(test())
"
```
Expected: No errors. Tables created.

**Step 3: Verify tables exist**

```bash
PGPASSWORD=nous_dev_password psql -h localhost -U nous -d nous -c "\dt heart.*"
```
Expected: `heart.subtasks` and `heart.schedules` in output.

**Step 4: Commit**

```bash
git add sql/migrations/010_subtasks.sql
git commit -m "feat(011.1): add subtasks and schedules tables"
```

---

## Task 3: ORM models for Subtask and Schedule

**Files:**
- Modify: `nous/storage/models.py` (append after ConversationState, ~line 537)

**Step 1: Write test for model instantiation**

Create `tests/test_subtasks.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_subtasks.py -v -x
```
Expected: ImportError — `Subtask` and `Schedule` not defined in models.py.

**Step 3: Add ORM models to `nous/storage/models.py`**

Append after the ConversationState class (after line 537):

```python
# ---------------------------------------------------------------------------
# 011.1: Subtasks & Scheduling (F009)
# ---------------------------------------------------------------------------


class Subtask(Base):
    """Background subtask queue entry."""

    __tablename__ = "subtasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="chk_subtask_status",
        ),
        {"schema": "heart"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False)
    parent_session_id: Mapped[str | None] = mapped_column(String(200))
    task: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default="100")
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    result: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(String(100))
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, server_default="120")
    notify: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )


class Schedule(Base):
    """Scheduled or recurring task."""

    __tablename__ = "schedules"
    __table_args__ = (
        CheckConstraint(
            "schedule_type IN ('once', 'recurring')",
            name="chk_schedule_type",
        ),
        {"schema": "heart"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    schedule_type: Mapped[str] = mapped_column(String(20), nullable=False)
    fire_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    interval_seconds: Mapped[int | None] = mapped_column(Integer)
    cron_expr: Mapped[str | None] = mapped_column(String(200))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_fire_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fire_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_fires: Mapped[int | None] = mapped_column(Integer)
    notify: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, server_default="120")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    created_by_session: Mapped[str | None] = mapped_column(String(200))
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_subtasks.py -v -x
```
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add nous/storage/models.py tests/test_subtasks.py
git commit -m "feat(011.1): add Subtask and Schedule ORM models"
```

---

## Task 4: SubtaskManager — CRUD and queue operations

**Files:**
- Create: `nous/heart/subtasks.py`
- Modify: `tests/test_subtasks.py`

**Step 1: Write failing tests for SubtaskManager**

Append to `tests/test_subtasks.py`:

```python
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
        s1 = await subtask_mgr.create(task="Task A")
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_subtasks.py::TestSubtaskManager -v -x
```
Expected: ImportError — `nous.heart.subtasks` doesn't exist.

**Step 3: Implement SubtaskManager**

Create `nous/heart/subtasks.py`:

```python
"""Subtask queue manager — CRUD and atomic dequeue for background tasks."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update

from nous.storage.database import Database
from nous.storage.models import Subtask

logger = logging.getLogger(__name__)

_PRIORITY_MAP = {"urgent": 50, "normal": 100, "low": 200}
_MAX_PENDING = 5


class SubtaskManager:
    """Manages the subtask queue in heart.subtasks."""

    def __init__(self, database: Database, agent_id: str) -> None:
        self._db = database
        self._agent_id = agent_id

    async def create(
        self,
        task: str,
        parent_session_id: str | None = None,
        priority: str = "normal",
        timeout: int = 120,
        notify: bool = True,
        metadata: dict | None = None,
    ) -> Subtask:
        """Create a new pending subtask. Raises ValueError if pending limit reached."""
        pri_val = _PRIORITY_MAP.get(priority, 100)

        async with self._db.session() as session:
            # Check pending limit
            count = await session.scalar(
                select(func.count())
                .select_from(Subtask)
                .where(Subtask.agent_id == self._agent_id)
                .where(Subtask.status == "pending")
            )
            if count >= _MAX_PENDING:
                raise ValueError(
                    f"pending subtask limit ({_MAX_PENDING}) reached"
                )

            subtask = Subtask(
                agent_id=self._agent_id,
                parent_session_id=parent_session_id,
                task=task,
                priority=pri_val,
                timeout_seconds=timeout,
                notify=notify,
                metadata_=metadata or {},
            )
            session.add(subtask)
            await session.commit()
            await session.refresh(subtask)
            logger.info("Created subtask %s: %s", subtask.id.hex[:8], task[:80])
            return subtask

    async def dequeue(self, worker_id: str) -> Subtask | None:
        """Atomically dequeue the highest-priority pending subtask."""
        async with self._db.session() as session:
            # SELECT FOR UPDATE SKIP LOCKED
            result = await session.execute(
                select(Subtask)
                .where(Subtask.agent_id == self._agent_id)
                .where(Subtask.status == "pending")
                .order_by(Subtask.priority, Subtask.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            subtask = result.scalar_one_or_none()
            if subtask is None:
                return None

            subtask.status = "running"
            subtask.worker_id = worker_id
            subtask.started_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(subtask)
            logger.info(
                "Dequeued subtask %s → %s", subtask.id.hex[:8], worker_id
            )
            return subtask

    async def complete(self, subtask_id: UUID, result: str) -> None:
        """Mark a subtask as completed with its result."""
        async with self._db.session() as session:
            await session.execute(
                update(Subtask)
                .where(Subtask.id == subtask_id)
                .values(
                    status="completed",
                    result=result,
                    completed_at=datetime.now(UTC),
                )
            )
            await session.commit()
            logger.info("Completed subtask %s", subtask_id.hex[:8])

    async def fail(self, subtask_id: UUID, error: str) -> None:
        """Mark a subtask as failed with error message."""
        async with self._db.session() as session:
            await session.execute(
                update(Subtask)
                .where(Subtask.id == subtask_id)
                .values(
                    status="failed",
                    error=error,
                    completed_at=datetime.now(UTC),
                )
            )
            await session.commit()
            logger.warning("Failed subtask %s: %s", subtask_id.hex[:8], error)

    async def cancel(self, subtask_id: UUID) -> bool:
        """Cancel a pending subtask. Returns False if not pending."""
        async with self._db.session() as session:
            result = await session.execute(
                update(Subtask)
                .where(Subtask.id == subtask_id)
                .where(Subtask.status == "pending")
                .values(status="cancelled", completed_at=datetime.now(UTC))
            )
            await session.commit()
            return result.rowcount > 0

    async def get(self, subtask_id: UUID) -> Subtask | None:
        """Get a subtask by ID."""
        async with self._db.session() as session:
            return await session.get(Subtask, subtask_id)

    async def list(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> list[Subtask]:
        """List subtasks, optionally filtered by status."""
        async with self._db.session() as session:
            q = (
                select(Subtask)
                .where(Subtask.agent_id == self._agent_id)
                .order_by(Subtask.created_at.desc())
                .limit(limit)
            )
            if status:
                q = q.where(Subtask.status == status)
            result = await session.execute(q)
            return list(result.scalars().all())

    async def reclaim_stale(self) -> int:
        """Re-enqueue running subtasks that exceeded their timeout."""
        async with self._db.session() as session:
            now = datetime.now(UTC)
            # Find running tasks past their timeout
            result = await session.execute(
                update(Subtask)
                .where(Subtask.agent_id == self._agent_id)
                .where(Subtask.status == "running")
                .where(
                    Subtask.started_at + func.make_interval(secs=Subtask.timeout_seconds) < now
                )
                .values(status="pending", worker_id=None, started_at=None)
            )
            await session.commit()
            if result.rowcount > 0:
                logger.warning("Reclaimed %d stale subtasks", result.rowcount)
            return result.rowcount

    async def count_by_status(self) -> dict[str, int]:
        """Count subtasks grouped by status."""
        async with self._db.session() as session:
            result = await session.execute(
                select(Subtask.status, func.count())
                .where(Subtask.agent_id == self._agent_id)
                .group_by(Subtask.status)
            )
            return dict(result.all())
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_subtasks.py -v -x
```
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add nous/heart/subtasks.py tests/test_subtasks.py
git commit -m "feat(011.1): SubtaskManager with queue operations"
```

---

## Task 5: ScheduleManager — CRUD and due-task operations

**Files:**
- Create: `nous/heart/schedules.py`
- Create: `tests/test_schedules.py`

**Step 1: Write failing tests**

Create `tests/test_schedules.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_schedules.py -v -x
```
Expected: ImportError.

**Step 3: Implement ScheduleManager**

Create `nous/heart/schedules.py`:

```python
"""Schedule manager — CRUD and due-task operations for recurring/timed tasks."""

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from croniter import croniter
from sqlalchemy import select, update

from nous.storage.database import Database
from nous.storage.models import Schedule

logger = logging.getLogger(__name__)


class ScheduleManager:
    """Manages scheduled tasks in heart.schedules."""

    def __init__(self, database: Database, agent_id: str) -> None:
        self._db = database
        self._agent_id = agent_id

    async def create(
        self,
        task: str,
        schedule_type: str,
        fire_at: datetime | None = None,
        interval_seconds: int | None = None,
        cron_expr: str | None = None,
        notify: bool = True,
        timeout: int = 120,
        max_fires: int | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> Schedule:
        """Create a new schedule."""
        # Compute next_fire_at
        now = datetime.now(UTC)
        if schedule_type == "once":
            next_fire = fire_at
        elif cron_expr:
            cron = croniter(cron_expr, now)
            next_fire = cron.get_next(datetime)
        elif interval_seconds:
            next_fire = now + timedelta(seconds=interval_seconds)
        else:
            raise ValueError("Recurring schedule needs interval_seconds or cron_expr")

        async with self._db.session() as session:
            schedule = Schedule(
                agent_id=self._agent_id,
                task=task,
                schedule_type=schedule_type,
                fire_at=fire_at,
                interval_seconds=interval_seconds,
                cron_expr=cron_expr,
                next_fire_at=next_fire,
                notify=notify,
                timeout_seconds=timeout,
                max_fires=max_fires,
                created_by_session=session_id,
                metadata_=metadata or {},
            )
            session.add(schedule)
            await session.commit()
            await session.refresh(schedule)
            logger.info(
                "Created %s schedule %s: %s (next: %s)",
                schedule_type, schedule.id.hex[:8], task[:80], next_fire,
            )
            return schedule

    async def get_due(self, now: datetime) -> list[Schedule]:
        """Get all active schedules whose next_fire_at <= now."""
        async with self._db.session() as session:
            result = await session.execute(
                select(Schedule)
                .where(Schedule.agent_id == self._agent_id)
                .where(Schedule.active.is_(True))
                .where(Schedule.next_fire_at <= now)
                .order_by(Schedule.next_fire_at)
            )
            return list(result.scalars().all())

    async def advance(self, schedule_id: UUID, fired_at: datetime) -> None:
        """Advance a recurring schedule after firing."""
        async with self._db.session() as session:
            schedule = await session.get(Schedule, schedule_id)
            if schedule is None:
                return

            schedule.fire_count += 1
            schedule.last_fired_at = fired_at

            # Check max_fires
            if schedule.max_fires and schedule.fire_count >= schedule.max_fires:
                schedule.active = False
                schedule.next_fire_at = None
            elif schedule.cron_expr:
                cron = croniter(schedule.cron_expr, fired_at)
                schedule.next_fire_at = cron.get_next(datetime)
            elif schedule.interval_seconds:
                schedule.next_fire_at = fired_at + timedelta(
                    seconds=schedule.interval_seconds
                )

            await session.commit()
            logger.info(
                "Advanced schedule %s (fire #%d, next: %s)",
                schedule_id.hex[:8], schedule.fire_count, schedule.next_fire_at,
            )

    async def deactivate(self, schedule_id: UUID) -> None:
        """Deactivate a schedule."""
        async with self._db.session() as session:
            await session.execute(
                update(Schedule)
                .where(Schedule.id == schedule_id)
                .values(active=False)
            )
            await session.commit()
            logger.info("Deactivated schedule %s", schedule_id.hex[:8])

    async def get(self, schedule_id: UUID) -> Schedule | None:
        """Get a schedule by ID."""
        async with self._db.session() as session:
            return await session.get(Schedule, schedule_id)

    async def list(self, active_only: bool = True, limit: int = 20) -> list[Schedule]:
        """List schedules, optionally filtered to active only."""
        async with self._db.session() as session:
            q = (
                select(Schedule)
                .where(Schedule.agent_id == self._agent_id)
                .order_by(Schedule.created_at.desc())
                .limit(limit)
            )
            if active_only:
                q = q.where(Schedule.active.is_(True))
            result = await session.execute(q)
            return list(result.scalars().all())
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_schedules.py -v -x
```
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add nous/heart/schedules.py tests/test_schedules.py
git commit -m "feat(011.1): ScheduleManager with due-task operations"
```

---

## Task 6: Wire managers into Heart

**Files:**
- Modify: `nous/heart/heart.py` (~line 75, after working_memory init)

**Step 1: Write test**

Append to `tests/test_subtasks.py`:

```python
from nous.heart.heart import Heart


class TestHeartIntegration:
    """Verify Heart exposes subtask and schedule managers."""

    async def test_heart_has_subtask_manager(self, db, settings):
        heart = Heart(db, settings)
        assert heart.subtasks is not None
        assert hasattr(heart.subtasks, "create")
        await heart.close()

    async def test_heart_has_schedule_manager(self, db, settings):
        heart = Heart(db, settings)
        assert heart.schedules is not None
        assert hasattr(heart.schedules, "create")
        await heart.close()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_subtasks.py::TestHeartIntegration -v -x
```
Expected: AttributeError — Heart has no `subtasks` attribute.

**Step 3: Add managers to Heart.__init__**

In `nous/heart/heart.py`, add imports at top:

```python
from nous.heart.subtasks import SubtaskManager
from nous.heart.schedules import ScheduleManager
```

In `__init__`, after `self.working_memory = WorkingMemoryManager(...)` add:

```python
        self.subtasks = SubtaskManager(database, settings.agent_id)
        self.schedules = ScheduleManager(database, settings.agent_id)
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_subtasks.py::TestHeartIntegration -v -x
```
Expected: PASS.

**Step 5: Commit**

```bash
git add nous/heart/heart.py tests/test_subtasks.py
git commit -m "feat(011.1): wire SubtaskManager and ScheduleManager into Heart"
```

---

## Task 7: Time parser

**Files:**
- Create: `nous/handlers/time_parser.py`
- Create: `tests/test_time_parser.py`

**Step 1: Write failing tests**

Create `tests/test_time_parser.py`:

```python
"""Tests for natural language time parsing (011.1)."""

from datetime import UTC, datetime, timedelta

import pytest

from nous.handlers.time_parser import parse_every, parse_when


class TestParseWhen:

    def test_iso_8601(self):
        result = parse_when("2026-03-10T09:00:00-05:00")
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 10

    def test_relative_hours(self):
        before = datetime.now(UTC)
        result = parse_when("in 2 hours")
        assert result > before
        assert result < before + timedelta(hours=2, minutes=1)

    def test_relative_minutes(self):
        before = datetime.now(UTC)
        result = parse_when("in 30 minutes")
        assert result > before
        assert result < before + timedelta(minutes=31)

    def test_relative_days(self):
        before = datetime.now(UTC)
        result = parse_when("in 3 days")
        assert result > before + timedelta(days=2)

    def test_rejects_past_time(self):
        with pytest.raises(ValueError, match="past"):
            parse_when("2020-01-01T00:00:00Z")

    def test_natural_language_tomorrow(self):
        result = parse_when("tomorrow 9am")
        assert result > datetime.now(UTC)


class TestParseEvery:

    def test_simple_minutes(self):
        interval, cron = parse_every("30 minutes")
        assert interval == 1800
        assert cron is None

    def test_simple_hours(self):
        interval, cron = parse_every("6 hours")
        assert interval == 21600
        assert cron is None

    def test_simple_days(self):
        interval, cron = parse_every("2 days")
        assert interval == 172800
        assert cron is None

    def test_daily_at_time(self):
        interval, cron = parse_every("daily at 8am")
        assert cron is not None
        assert interval is None

    def test_daily_at_time_with_timezone(self):
        interval, cron = parse_every("daily at 9am EST")
        assert cron is not None

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_every("whenever you feel like it")
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_time_parser.py -v -x
```
Expected: ImportError.

**Step 3: Implement time parser**

Create `nous/handlers/time_parser.py`:

```python
"""Natural language time parsing for schedule_task tool."""

import re
from datetime import UTC, datetime, timedelta

from croniter import croniter
from dateutil import parser as dateutil_parser

# Relative time: "in N hours/minutes/days/weeks"
_RELATIVE_RE = re.compile(
    r"in\s+(\d+)\s+(minute|hour|day|week)s?", re.IGNORECASE
)

_UNIT_SECONDS = {
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
}

# Simple interval: "N hours/minutes/days"
_INTERVAL_RE = re.compile(
    r"(\d+)\s+(minute|hour|day|week)s?", re.IGNORECASE
)

# "daily at 8am", "daily at 9am EST"
_DAILY_RE = re.compile(
    r"daily\s+at\s+(\d{1,2})\s*(am|pm)\s*([\w/]+)?", re.IGNORECASE
)

# "every monday at 10am"
_WEEKLY_RE = re.compile(
    r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})\s*(am|pm)",
    re.IGNORECASE,
)

_DOW_MAP = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 0,
}


def parse_when(when: str) -> datetime:
    """Parse a 'when' string into an aware UTC datetime.

    Supports:
    - ISO 8601: "2026-03-10T09:00:00-05:00"
    - Relative: "in 2 hours", "in 30 minutes"
    - Natural: "tomorrow 9am", "next monday 8am EST"

    Raises ValueError if time is in the past or unparseable.
    """
    now = datetime.now(UTC)

    # Try relative first
    m = _RELATIVE_RE.match(when.strip())
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        delta = timedelta(seconds=amount * _UNIT_SECONDS[unit])
        return now + delta

    # Try dateutil for ISO and natural language
    try:
        result = dateutil_parser.parse(when, fuzzy=True)
        # Make aware if naive
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        else:
            result = result.astimezone(UTC)

        if result < now:
            raise ValueError(f"Scheduled time is in the past: {result.isoformat()}")
        return result
    except (ValueError, OverflowError) as e:
        if "past" in str(e):
            raise
        raise ValueError(f"Cannot parse time: '{when}'") from e


def parse_every(every: str) -> tuple[int | None, str | None]:
    """Parse an 'every' string into (interval_seconds, cron_expr).

    Returns one or both:
    - Simple intervals ("30 minutes") → (1800, None)
    - Daily patterns ("daily at 8am") → (None, "0 8 * * *")
    - Weekly patterns ("every monday at 10am") → (None, "0 10 * * 1")

    Raises ValueError if unparseable.
    """
    text = every.strip()

    # Daily pattern: "daily at 8am [TZ]"
    m = _DAILY_RE.match(text)
    if m:
        hour = int(m.group(1))
        ampm = m.group(2).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        cron = f"0 {hour} * * *"
        # Validate with croniter
        croniter(cron)
        return None, cron

    # Weekly pattern: "every monday at 10am"
    m = _WEEKLY_RE.match(text)
    if m:
        dow = _DOW_MAP[m.group(1).lower()]
        hour = int(m.group(2))
        ampm = m.group(3).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        cron = f"0 {hour} * * {dow}"
        croniter(cron)
        return None, cron

    # Simple interval: "N units"
    m = _INTERVAL_RE.match(text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        return amount * _UNIT_SECONDS[unit], None

    raise ValueError(f"Cannot parse recurring schedule: '{every}'")
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_time_parser.py -v -x
```
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add nous/handlers/time_parser.py tests/test_time_parser.py
git commit -m "feat(011.1): time parser for schedule_task"
```

---

## Task 8: Configuration settings

**Files:**
- Modify: `nous/config.py` (add after line 144, before validators)

**Step 1: Add settings**

In `nous/config.py`, before the `@model_validator` blocks (line 146), add:

```python
    # 011.1: Subtasks & Scheduling
    subtask_enabled: bool = True
    subtask_workers: int = 2
    subtask_poll_interval: float = 2.0
    subtask_default_timeout: int = 120
    subtask_max_timeout: int = 600
    subtask_max_concurrent: int = 3
    schedule_enabled: bool = True
    schedule_check_interval: int = 60
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
```

**Step 2: Verify config loads**

```bash
uv run python -c "from nous.config import Settings; s = Settings(); print(f'subtask_enabled={s.subtask_enabled}, workers={s.subtask_workers}')"
```
Expected: `subtask_enabled=True, workers=2`

**Step 3: Commit**

```bash
git add nous/config.py
git commit -m "feat(011.1): add subtask and schedule config settings"
```

---

## Task 9: SubtaskWorkerPool

**Files:**
- Create: `nous/handlers/subtask_worker.py`
- Modify: `tests/test_subtasks.py` (add worker pool tests)

**Step 1: Write failing tests**

Append to `tests/test_subtasks.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from nous.handlers.subtask_worker import SubtaskWorkerPool


class TestSubtaskWorkerPool:
    """Worker pool execution tests."""

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
        bus = MagicMock()
        bus.emit = MagicMock()
        return bus

    @pytest.fixture
    def mock_http(self):
        return AsyncMock()

    async def test_worker_executes_subtask(
        self, db, settings, mock_runner, mock_bus, mock_http,
    ):
        heart = Heart(db, settings)
        pool = SubtaskWorkerPool(
            runner=mock_runner, heart=heart,
            settings=settings, bus=mock_bus, http_client=mock_http,
        )

        # Create a subtask
        subtask = await heart.subtasks.create(task="Test execution")

        # Execute directly (not via worker loop)
        result = await pool._execute_subtask(subtask)
        assert result == "Task completed successfully"
        mock_runner.run_turn.assert_called_once()

        # Verify session_id format
        call_kwargs = mock_runner.run_turn.call_args
        assert call_kwargs[1]["session_id"].startswith("subtask-")

        await heart.close()

    async def test_worker_handles_timeout(
        self, db, settings, mock_runner, mock_bus, mock_http,
    ):
        heart = Heart(db, settings)

        # Make runner hang
        async def slow_turn(**kwargs):
            await asyncio.sleep(10)
            return ("done", MagicMock(), {})
        mock_runner.run_turn = slow_turn

        pool = SubtaskWorkerPool(
            runner=mock_runner, heart=heart,
            settings=settings, bus=mock_bus, http_client=mock_http,
        )

        subtask = await heart.subtasks.create(task="Slow task", timeout=1)
        dequeued = await heart.subtasks.dequeue("test-worker")

        await pool._process_subtask(dequeued)

        updated = await heart.subtasks.get(subtask.id)
        assert updated.status == "failed"
        assert "Timeout" in (updated.error or "")

        await heart.close()
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_subtasks.py::TestSubtaskWorkerPool -v -x
```
Expected: ImportError.

**Step 3: Implement SubtaskWorkerPool**

Create `nous/handlers/subtask_worker.py`:

```python
"""Background worker pool for executing subtasks as agent turns."""

import asyncio
import logging

import httpx

from nous.config import Settings
from nous.events import Event, EventBus
from nous.heart.heart import Heart
from nous.storage.models import Subtask

logger = logging.getLogger(__name__)


class SubtaskWorkerPool:
    """Pool of async workers that execute subtasks via AgentRunner."""

    def __init__(
        self,
        runner: object,  # AgentRunner — typed as object to avoid circular import
        heart: Heart,
        settings: Settings,
        bus: EventBus | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._runner = runner
        self._heart = heart
        self._settings = settings
        self._bus = bus
        self._http = http_client
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start worker tasks and reclaim any stale subtasks."""
        reclaimed = await self._heart.subtasks.reclaim_stale()
        if reclaimed:
            logger.info("Reclaimed %d stale subtasks on startup", reclaimed)

        for i in range(self._settings.subtask_workers):
            task = asyncio.create_task(
                self._worker_loop(f"worker-{i}"),
                name=f"subtask-worker-{i}",
            )
            self._workers.append(task)
        logger.info("Started %d subtask workers", len(self._workers))

    async def stop(self) -> None:
        """Cancel all workers."""
        for w in self._workers:
            w.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Stopped subtask workers")

    async def _worker_loop(self, worker_id: str) -> None:
        """Poll for subtasks and execute them."""
        logger.info("Subtask %s started", worker_id)
        while True:
            try:
                subtask = await self._heart.subtasks.dequeue(worker_id)
                if subtask is None:
                    await asyncio.sleep(self._settings.subtask_poll_interval)
                    continue

                # Check concurrency limit
                running = await self._heart.subtasks.count_by_status()
                if running.get("running", 0) > self._settings.subtask_max_concurrent:
                    # Put it back — re-enqueue by failing then creating new
                    logger.warning("Concurrency limit hit, re-sleeping")
                    await asyncio.sleep(self._settings.subtask_poll_interval)
                    continue

                await self._process_subtask(subtask)

            except asyncio.CancelledError:
                logger.info("Subtask %s cancelled", worker_id)
                return
            except Exception:
                logger.exception("Subtask %s error", worker_id)
                await asyncio.sleep(self._settings.subtask_poll_interval)

    async def _process_subtask(self, subtask: Subtask) -> None:
        """Execute a single subtask with timeout handling."""
        try:
            result = await asyncio.wait_for(
                self._execute_subtask(subtask),
                timeout=subtask.timeout_seconds,
            )
            await self._heart.subtasks.complete(subtask.id, result)
            self._emit_event("subtask_completed", subtask, result=result)
            if subtask.notify:
                await self._send_telegram(subtask, result, success=True)
            logger.info(
                "Subtask %s completed: %s",
                subtask.id.hex[:8], subtask.task[:60],
            )

        except asyncio.TimeoutError:
            error = f"Timeout after {subtask.timeout_seconds}s"
            await self._heart.subtasks.fail(subtask.id, error)
            self._emit_event("subtask_failed", subtask, error=error)
            if subtask.notify:
                await self._send_telegram(subtask, error, success=False)
            logger.warning("Subtask %s timed out", subtask.id.hex[:8])

        except Exception as e:
            error = str(e)
            await self._heart.subtasks.fail(subtask.id, error)
            self._emit_event("subtask_failed", subtask, error=error)
            if subtask.notify:
                await self._send_telegram(subtask, error, success=False)
            logger.exception("Subtask %s failed", subtask.id.hex[:8])

    async def _execute_subtask(self, subtask: Subtask) -> str:
        """Execute a subtask as an independent agent session."""
        session_id = f"subtask-{subtask.id.hex[:8]}"

        system_prefix = (
            f"You are executing a background subtask.\n"
            f"Task: {subtask.task}\n"
            f"Parent session: {subtask.parent_session_id or 'none'}\n"
            f"Deliver a clear, complete result. Do not ask questions.\n"
        )

        response_text, _ctx, _usage = await self._runner.run_turn(
            session_id=session_id,
            user_message=subtask.task,
            agent_id=self._settings.agent_id,
            system_prompt_prefix=system_prefix,
        )
        return response_text

    def _emit_event(
        self,
        event_type: str,
        subtask: Subtask,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """Emit an event on the bus if available."""
        if self._bus is None:
            return
        data = {
            "subtask_id": str(subtask.id),
            "task": subtask.task,
            "parent_session": subtask.parent_session_id,
        }
        if result:
            data["result"] = result[:2000]
        if error:
            data["error"] = error
        self._bus.emit(Event(
            type=event_type,
            agent_id=subtask.agent_id,
            session_id=subtask.parent_session_id,
            data=data,
        ))

    async def _send_telegram(
        self, subtask: Subtask, text: str, *, success: bool
    ) -> None:
        """Send a Telegram notification about subtask completion."""
        token = self._settings.telegram_bot_token
        chat_id = self._settings.telegram_chat_id
        if not token or not chat_id or not self._http:
            return

        emoji = "\u2705" if success else "\u274c"
        label = "completed" if success else "failed"
        task_preview = subtask.task[:200]
        result_preview = text[:3800]

        message = f"{emoji} Task {label}: \"{task_preview}\"\n\n{result_preview}"

        try:
            await self._http.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
                timeout=10,
            )
        except Exception:
            logger.warning("Failed to send Telegram notification for subtask %s", subtask.id.hex[:8])
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_subtasks.py::TestSubtaskWorkerPool -v -x
```
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add nous/handlers/subtask_worker.py tests/test_subtasks.py
git commit -m "feat(011.1): SubtaskWorkerPool with execution and notifications"
```

---

## Task 10: TaskScheduler

**Files:**
- Create: `nous/handlers/task_scheduler.py`
- Modify: `tests/test_schedules.py` (add scheduler tests)

**Step 1: Write failing tests**

Append to `tests/test_schedules.py`:

```python
from nous.handlers.task_scheduler import TaskScheduler
from nous.heart.heart import Heart


class TestTaskScheduler:

    async def test_fires_due_once_schedule(self, db, settings):
        heart = Heart(db, settings)
        scheduler = TaskScheduler(heart, settings)

        # Create a schedule due now
        past = datetime.now(UTC) - timedelta(minutes=1)
        schedule = await heart.schedules.create(
            task="Due task",
            schedule_type="once",
            fire_at=past,
        )

        # Fire due tasks
        fired = await scheduler._fire_due_tasks()
        assert fired == 1

        # Schedule should be deactivated
        updated = await heart.schedules.get(schedule.id)
        assert updated.active is False

        # Subtask should be created
        subtasks = await heart.subtasks.list(limit=10)
        assert len(subtasks) == 1
        assert subtasks[0].task == "Due task"

        await heart.close()

    async def test_fires_and_advances_recurring(self, db, settings):
        heart = Heart(db, settings)
        scheduler = TaskScheduler(heart, settings)

        # Create recurring schedule due now
        schedule = await heart.schedules.create(
            task="Recurring task",
            schedule_type="recurring",
            interval_seconds=3600,
        )
        # Backdate next_fire_at to make it due
        async with heart.db.session() as sess:
            from sqlalchemy import update as sql_update
            from nous.storage.models import Schedule as ScheduleModel
            await sess.execute(
                sql_update(ScheduleModel)
                .where(ScheduleModel.id == schedule.id)
                .values(next_fire_at=datetime.now(UTC) - timedelta(minutes=1))
            )
            await sess.commit()

        fired = await scheduler._fire_due_tasks()
        assert fired == 1

        # Schedule should still be active with advanced next_fire_at
        updated = await heart.schedules.get(schedule.id)
        assert updated.active is True
        assert updated.fire_count == 1
        assert updated.next_fire_at > datetime.now(UTC)

        await heart.close()

    async def test_skips_future_schedules(self, db, settings):
        heart = Heart(db, settings)
        scheduler = TaskScheduler(heart, settings)

        future = datetime.now(UTC) + timedelta(hours=5)
        await heart.schedules.create(
            task="Future task",
            schedule_type="once",
            fire_at=future,
        )

        fired = await scheduler._fire_due_tasks()
        assert fired == 0

        await heart.close()
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_schedules.py::TestTaskScheduler -v -x
```
Expected: ImportError.

**Step 3: Implement TaskScheduler**

Create `nous/handlers/task_scheduler.py`:

```python
"""Background scheduler that enqueues due tasks as subtasks."""

import asyncio
import logging
from datetime import UTC, datetime

from nous.config import Settings
from nous.heart.heart import Heart

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Periodically checks for due schedules and creates subtasks."""

    def __init__(self, heart: Heart, settings: Settings) -> None:
        self._heart = heart
        self._settings = settings
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the scheduler check loop."""
        self._task = asyncio.create_task(
            self._loop(), name="task-scheduler"
        )
        logger.info(
            "TaskScheduler started (check every %ds)",
            self._settings.schedule_check_interval,
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("TaskScheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop."""
        while True:
            try:
                await asyncio.sleep(self._settings.schedule_check_interval)
                await self._fire_due_tasks()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Scheduler loop error")

    async def _fire_due_tasks(self) -> int:
        """Find and enqueue all due scheduled tasks. Returns count fired."""
        now = datetime.now(UTC)
        due = await self._heart.schedules.get_due(now)
        if not due:
            return 0

        fired = 0
        for schedule in due:
            try:
                await self._heart.subtasks.create(
                    task=schedule.task,
                    parent_session_id=f"schedule-{schedule.id.hex[:8]}",
                    timeout=schedule.timeout_seconds,
                    notify=schedule.notify,
                )

                if schedule.schedule_type == "once":
                    await self._heart.schedules.deactivate(schedule.id)
                else:
                    await self._heart.schedules.advance(schedule.id, now)

                fired += 1
                logger.info(
                    "Fired schedule %s: %s",
                    schedule.id.hex[:8], schedule.task[:60],
                )
            except ValueError:
                logger.warning(
                    "Couldn't enqueue schedule %s (queue full?)",
                    schedule.id.hex[:8],
                )
            except Exception:
                logger.exception(
                    "Error firing schedule %s", schedule.id.hex[:8]
                )

        return fired
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_schedules.py -v -x
```
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add nous/handlers/task_scheduler.py tests/test_schedules.py
git commit -m "feat(011.1): TaskScheduler for due-task enqueuing"
```

---

## Task 11: Agent tools — spawn_task, schedule_task, list_tasks, cancel_task

**Files:**
- Modify: `nous/api/tools.py` (add schemas + handlers + registration)
- Modify: `nous/api/runner.py` (update FRAME_TOOLS)

**Step 1: Add tool schemas and handlers to `nous/api/tools.py`**

After the existing `register_nous_tools` function (line 612), add:

```python
# ---------------------------------------------------------------------------
# 011.1: Subtask & Schedule tools
# ---------------------------------------------------------------------------

_SPAWN_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Spawn a background subtask for parallel or long-running work. "
        "The subtask runs as a full agent turn with all tools. "
        "Results are delivered via Telegram and stored in memory."
    ),
    "properties": {
        "task": {
            "type": "string",
            "description": "Clear, specific instruction for what to do",
        },
        "priority": {
            "type": "string",
            "description": "Task priority",
            "enum": ["urgent", "normal", "low"],
            "default": "normal",
        },
        "timeout": {
            "type": "integer",
            "description": "Max seconds (default 120, max 600)",
            "default": 120,
            "minimum": 10,
            "maximum": 600,
        },
        "notify": {
            "type": "boolean",
            "description": "Send Telegram notification when done",
            "default": True,
        },
    },
    "required": ["task"],
}

_SCHEDULE_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Schedule a task for later or recurring execution. "
        "Use 'when' for one-shot timers, 'every' for recurring. "
        "Exactly one of 'when' or 'every' must be provided."
    ),
    "properties": {
        "task": {
            "type": "string",
            "description": "What to do when the schedule fires",
        },
        "when": {
            "type": "string",
            "description": "One-shot: ISO timestamp or relative ('in 2 hours', 'tomorrow 9am')",
        },
        "every": {
            "type": "string",
            "description": "Recurring: interval ('30 minutes', '6 hours', 'daily at 9am EST')",
        },
        "notify": {
            "type": "boolean",
            "description": "Send Telegram notification with results",
            "default": True,
        },
    },
    "required": ["task"],
}

_LIST_TASKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "List subtasks and scheduled tasks.",
    "properties": {
        "status": {
            "type": "string",
            "description": "Filter: pending, running, completed, scheduled, all",
            "enum": ["pending", "running", "completed", "scheduled", "all"],
            "default": "all",
        },
    },
    "required": [],
}

_CANCEL_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Cancel a pending subtask or deactivate a scheduled task.",
    "properties": {
        "task_id": {
            "type": "string",
            "description": "UUID of the subtask or schedule to cancel",
        },
    },
    "required": ["task_id"],
}


def create_subtask_tools(
    heart: "Heart", settings: "Settings",
) -> dict[str, Any]:
    """Create closure-based subtask/schedule tool handlers."""
    from nous.handlers.time_parser import parse_every, parse_when

    async def spawn_task(
        task: str,
        priority: str = "normal",
        timeout: int = 120,
        notify: bool = True,
    ) -> dict:
        timeout = min(timeout, settings.subtask_max_timeout)
        try:
            subtask = await heart.subtasks.create(
                task=task,
                priority=priority,
                timeout=timeout,
                notify=notify,
            )
            return {"content": [{"type": "text", "text": (
                f"Subtask spawned: {subtask.id.hex[:8]}\n"
                f"Task: {task[:200]}\n"
                f"Priority: {priority}, Timeout: {timeout}s"
            )}]}
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"Cannot spawn: {e}"}]}

    async def schedule_task(
        task: str,
        when: str | None = None,
        every: str | None = None,
        notify: bool = True,
    ) -> dict:
        if when and every:
            return {"content": [{"type": "text", "text":
                "Provide exactly one of 'when' or 'every', not both."}]}
        if not when and not every:
            return {"content": [{"type": "text", "text":
                "Provide either 'when' (one-shot) or 'every' (recurring)."}]}

        try:
            if when:
                fire_at = parse_when(when)
                schedule = await heart.schedules.create(
                    task=task, schedule_type="once",
                    fire_at=fire_at, notify=notify,
                )
                return {"content": [{"type": "text", "text": (
                    f"Scheduled: {schedule.id.hex[:8]}\n"
                    f"Task: {task[:200]}\n"
                    f"Fires at: {fire_at.isoformat()}"
                )}]}
            else:
                interval, cron = parse_every(every)
                schedule = await heart.schedules.create(
                    task=task, schedule_type="recurring",
                    interval_seconds=interval, cron_expr=cron,
                    notify=notify,
                )
                return {"content": [{"type": "text", "text": (
                    f"Recurring schedule: {schedule.id.hex[:8]}\n"
                    f"Task: {task[:200]}\n"
                    f"Next fire: {schedule.next_fire_at.isoformat()}"
                )}]}
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"Schedule error: {e}"}]}

    async def list_tasks(status: str = "all") -> dict:
        lines = []

        if status in ("all", "pending", "running", "completed"):
            subtasks = await heart.subtasks.list(
                status=None if status == "all" else status, limit=20,
            )
            for st in subtasks:
                lines.append(
                    f"[subtask] {st.id.hex[:8]} | {st.status} | {st.task[:60]}"
                )

        if status in ("all", "scheduled"):
            schedules = await heart.schedules.list(active_only=(status == "scheduled"))
            for sc in schedules:
                next_str = sc.next_fire_at.isoformat() if sc.next_fire_at else "n/a"
                lines.append(
                    f"[schedule] {sc.id.hex[:8]} | {sc.schedule_type} | "
                    f"next: {next_str} | {sc.task[:60]}"
                )

        if not lines:
            return {"content": [{"type": "text", "text": "No tasks found."}]}
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    async def cancel_task(task_id: str) -> dict:
        from uuid import UUID
        try:
            uid = UUID(task_id)
        except ValueError:
            return {"content": [{"type": "text", "text": f"Invalid UUID: {task_id}"}]}

        # Try subtask first
        if await heart.subtasks.cancel(uid):
            return {"content": [{"type": "text", "text": f"Cancelled subtask {task_id[:8]}"}]}

        # Try schedule
        schedule = await heart.schedules.get(uid)
        if schedule and schedule.active:
            await heart.schedules.deactivate(uid)
            return {"content": [{"type": "text", "text": f"Deactivated schedule {task_id[:8]}"}]}

        return {"content": [{"type": "text", "text": f"No active task found: {task_id[:8]}"}]}

    return {
        "spawn_task": spawn_task,
        "schedule_task": schedule_task,
        "list_tasks": list_tasks,
        "cancel_task": cancel_task,
    }


def register_subtask_tools(
    dispatcher: "ToolDispatcher", heart: "Heart", settings: "Settings",
) -> None:
    """Register subtask/schedule tools with the dispatcher."""
    closures = create_subtask_tools(heart, settings)
    dispatcher.register("spawn_task", closures["spawn_task"], _SPAWN_TASK_SCHEMA)
    dispatcher.register("schedule_task", closures["schedule_task"], _SCHEDULE_TASK_SCHEMA)
    dispatcher.register("list_tasks", closures["list_tasks"], _LIST_TASKS_SCHEMA)
    dispatcher.register("cancel_task", closures["cancel_task"], _CANCEL_TASK_SCHEMA)
```

**Step 2: Update FRAME_TOOLS in `nous/api/runner.py`**

Add `spawn_task`, `schedule_task` to `conversation`, `task`, `debug` frames. Add `list_tasks`, `cancel_task` to all frames.

In `nous/api/runner.py` around line 38-46, update the FRAME_TOOLS dict:

```python
FRAME_TOOLS: dict[str, list[str]] = {
    "conversation": ["record_decision", "learn_fact", "recall_deep", "recall_recent", "create_censor", "bash", "read_file", "write_file", "web_search", "web_fetch", "spawn_task", "schedule_task", "list_tasks", "cancel_task"],
    "question": ["recall_deep", "recall_recent", "bash", "read_file", "write_file", "record_decision", "learn_fact", "create_censor", "web_search", "web_fetch", "list_tasks", "cancel_task"],
    "decision": ["record_decision", "recall_deep", "recall_recent", "create_censor", "bash", "read_file", "web_search", "web_fetch", "list_tasks", "cancel_task"],
    "creative": ["learn_fact", "recall_deep", "recall_recent", "write_file", "web_search", "list_tasks"],
    "task": ["*"],  # All tools
    "debug": ["record_decision", "recall_deep", "recall_recent", "bash", "read_file", "learn_fact", "web_search", "web_fetch", "spawn_task", "schedule_task", "list_tasks", "cancel_task"],
    "initiation": ["store_identity", "complete_initiation"],
}
```

**Step 3: Run existing tests to verify nothing broke**

```bash
uv run pytest tests/ -v -x --timeout=60 -q
```
Expected: All existing tests still PASS.

**Step 4: Commit**

```bash
git add nous/api/tools.py nous/api/runner.py
git commit -m "feat(011.1): spawn_task, schedule_task, list_tasks, cancel_task tools"
```

---

## Task 12: REST endpoints for subtasks and schedules

**Files:**
- Modify: `nous/api/rest.py` (add 6 endpoints before routes list)

**Step 1: Add endpoints to `nous/api/rest.py`**

Before the `routes = [` line (~line 439), add the subtask/schedule endpoints:

```python
    # ------------------------------------------------------------------
    # 011.1: Subtask & Schedule endpoints
    # ------------------------------------------------------------------

    async def list_subtasks(request: Request) -> JSONResponse:
        """GET /subtasks?status=all&limit=20"""
        status = request.query_params.get("status")
        if status == "all":
            status = None
        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)

        try:
            subtasks = await heart.subtasks.list(status=status, limit=limit)
            return JSONResponse({
                "subtasks": [
                    {
                        "id": str(s.id),
                        "task": s.task,
                        "status": s.status,
                        "priority": s.priority,
                        "result": s.result,
                        "error": s.error,
                        "created_at": s.created_at.isoformat() if s.created_at else None,
                        "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    }
                    for s in subtasks
                ],
            })
        except Exception as e:
            logger.error("List subtasks error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def get_subtask(request: Request) -> JSONResponse:
        """GET /subtasks/{id}"""
        try:
            subtask_id = UUID(request.path_params["id"])
        except ValueError:
            return JSONResponse({"error": "Invalid subtask ID"}, status_code=400)

        subtask = await heart.subtasks.get(subtask_id)
        if subtask is None:
            return JSONResponse({"error": "Subtask not found"}, status_code=404)

        return JSONResponse({
            "id": str(subtask.id),
            "task": subtask.task,
            "status": subtask.status,
            "priority": subtask.priority,
            "result": subtask.result,
            "error": subtask.error,
            "parent_session_id": subtask.parent_session_id,
            "worker_id": subtask.worker_id,
            "timeout_seconds": subtask.timeout_seconds,
            "created_at": subtask.created_at.isoformat() if subtask.created_at else None,
            "started_at": subtask.started_at.isoformat() if subtask.started_at else None,
            "completed_at": subtask.completed_at.isoformat() if subtask.completed_at else None,
        })

    async def cancel_subtask(request: Request) -> JSONResponse:
        """DELETE /subtasks/{id}"""
        try:
            subtask_id = UUID(request.path_params["id"])
        except ValueError:
            return JSONResponse({"error": "Invalid subtask ID"}, status_code=400)

        cancelled = await heart.subtasks.cancel(subtask_id)
        if not cancelled:
            return JSONResponse({"error": "Cannot cancel (not pending or not found)"}, status_code=409)
        return JSONResponse({"status": "cancelled", "id": str(subtask_id)})

    async def list_schedules_endpoint(request: Request) -> JSONResponse:
        """GET /schedules?active_only=true"""
        active_only = request.query_params.get("active_only", "true").lower() == "true"

        try:
            schedules = await heart.schedules.list(active_only=active_only)
            return JSONResponse({
                "schedules": [
                    {
                        "id": str(s.id),
                        "task": s.task,
                        "schedule_type": s.schedule_type,
                        "active": s.active,
                        "next_fire_at": s.next_fire_at.isoformat() if s.next_fire_at else None,
                        "fire_count": s.fire_count,
                        "max_fires": s.max_fires,
                        "cron_expr": s.cron_expr,
                        "interval_seconds": s.interval_seconds,
                        "created_at": s.created_at.isoformat() if s.created_at else None,
                    }
                    for s in schedules
                ],
            })
        except Exception as e:
            logger.error("List schedules error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def create_schedule_endpoint(request: Request) -> JSONResponse:
        """POST /schedules"""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        task = body.get("task")
        if not task:
            return JSONResponse({"error": "Missing 'task' field"}, status_code=400)

        when = body.get("when")
        every = body.get("every")
        notify = body.get("notify", True)

        if when and every:
            return JSONResponse({"error": "Provide 'when' or 'every', not both"}, status_code=400)
        if not when and not every:
            return JSONResponse({"error": "Provide 'when' or 'every'"}, status_code=400)

        try:
            from nous.handlers.time_parser import parse_every, parse_when

            if when:
                fire_at = parse_when(when)
                schedule = await heart.schedules.create(
                    task=task, schedule_type="once",
                    fire_at=fire_at, notify=notify,
                )
            else:
                interval, cron = parse_every(every)
                schedule = await heart.schedules.create(
                    task=task, schedule_type="recurring",
                    interval_seconds=interval, cron_expr=cron,
                    notify=notify,
                )

            return JSONResponse({
                "id": str(schedule.id),
                "task": schedule.task,
                "schedule_type": schedule.schedule_type,
                "next_fire_at": schedule.next_fire_at.isoformat() if schedule.next_fire_at else None,
            }, status_code=201)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.error("Create schedule error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def delete_schedule(request: Request) -> JSONResponse:
        """DELETE /schedules/{id}"""
        try:
            schedule_id = UUID(request.path_params["id"])
        except ValueError:
            return JSONResponse({"error": "Invalid schedule ID"}, status_code=400)

        schedule = await heart.schedules.get(schedule_id)
        if schedule is None:
            return JSONResponse({"error": "Schedule not found"}, status_code=404)

        await heart.schedules.deactivate(schedule_id)
        return JSONResponse({"status": "deactivated", "id": str(schedule_id)})
```

Then add the new routes to the `routes` list:

```python
    routes = [
        # ... existing routes ...
        Route("/subtasks", list_subtasks),
        Route("/subtasks/{id}", get_subtask),
        Route("/subtasks/{id}", cancel_subtask, methods=["DELETE"]),
        Route("/schedules", list_schedules_endpoint),
        Route("/schedules", create_schedule_endpoint, methods=["POST"]),
        Route("/schedules/{id}", delete_schedule, methods=["DELETE"]),
        Route("/health", health),
    ]
```

**Step 2: Run all tests**

```bash
uv run pytest tests/ -v -x --timeout=60 -q
```
Expected: All tests PASS.

**Step 3: Commit**

```bash
git add nous/api/rest.py
git commit -m "feat(011.1): REST endpoints for subtasks and schedules"
```

---

## Task 13: Startup wiring in main.py

**Files:**
- Modify: `nous/main.py`

**Step 1: Add worker pool and scheduler to `create_components`**

In `nous/main.py`, after the runner is created and started (~line 175), add:

```python
    # 011.1: Subtask worker pool
    subtask_pool = None
    if settings.subtask_enabled and bus is not None:
        try:
            from nous.handlers.subtask_worker import SubtaskWorkerPool

            subtask_pool = SubtaskWorkerPool(
                runner=runner, heart=heart, settings=settings,
                bus=bus, http_client=handler_http,
            )
            await subtask_pool.start()
        except ImportError:
            logger.debug("SubtaskWorkerPool not available yet")

    # 011.1: Task scheduler
    task_scheduler = None
    if settings.schedule_enabled:
        try:
            from nous.handlers.task_scheduler import TaskScheduler

            task_scheduler = TaskScheduler(heart, settings)
            await task_scheduler.start()
        except ImportError:
            logger.debug("TaskScheduler not available yet")
```

Also register subtask tools after existing tool registration (~after line 171):

```python
    # 011.1: Subtask/schedule tools
    if settings.subtask_enabled:
        from nous.api.tools import register_subtask_tools
        register_subtask_tools(dispatcher, heart, settings)
```

Add the new components to the return dict:

```python
    return {
        # ... existing entries ...
        "subtask_pool": subtask_pool,
        "task_scheduler": task_scheduler,
    }
```

**Step 2: Add shutdown for new components**

In `shutdown_components`, add before the session_monitor stop:

```python
    # 011.1: Stop subtask pool and scheduler first
    subtask_pool = components.get("subtask_pool")
    if subtask_pool:
        await subtask_pool.stop()

    task_scheduler = components.get("task_scheduler")
    if task_scheduler:
        await task_scheduler.stop()
```

**Step 3: Verify startup works**

```bash
uv run python -c "
import asyncio
from nous.config import Settings
from nous.main import create_components, shutdown_components
async def test():
    s = Settings()
    s.subtask_enabled = True
    s.schedule_enabled = True
    c = await create_components(s)
    print('subtask_pool:', c.get('subtask_pool'))
    print('task_scheduler:', c.get('task_scheduler'))
    await shutdown_components(c)
asyncio.run(test())
"
```
Expected: Components created and shutdown cleanly.

**Step 4: Run full test suite**

```bash
uv run pytest tests/ -v --timeout=60 -q
```
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add nous/main.py
git commit -m "feat(011.1): wire worker pool and scheduler into startup"
```

---

## Task 14: AgentRunner — system_prompt_prefix support for subtasks

**Files:**
- Modify: `nous/api/runner.py`

**Step 1: Check if `run_turn` already supports `system_prompt_prefix`**

Read `nous/api/runner.py` around line 268. If it doesn't have `system_prompt_prefix` parameter, add it.

Add `system_prompt_prefix: str | None = None` to the `run_turn` signature. In the system prompt assembly (where the cognitive context is built), prepend the prefix if provided.

This is a small change — just one new parameter and a string concatenation.

**Step 2: Run full tests**

```bash
uv run pytest tests/ -v --timeout=60 -q
```
Expected: All tests PASS (no existing tests use this parameter).

**Step 3: Commit**

```bash
git add nous/api/runner.py
git commit -m "feat(011.1): add system_prompt_prefix to run_turn for subtask scoping"
```

---

## Task 15: Final integration test

**Files:**
- Modify: `tests/test_subtasks.py` (add end-to-end integration test)

**Step 1: Write integration test**

Append to `tests/test_subtasks.py`:

```python
class TestIntegration:
    """End-to-end integration tests."""

    async def test_full_subtask_lifecycle(self, db, settings):
        """Create → dequeue → complete → verify."""
        heart = Heart(db, settings)

        # Create
        subtask = await heart.subtasks.create(
            task="Integration test task",
            priority="urgent",
            timeout=60,
        )
        assert subtask.status == "pending"

        # Dequeue
        dequeued = await heart.subtasks.dequeue("test-worker")
        assert dequeued is not None
        assert dequeued.id == subtask.id
        assert dequeued.status == "running"

        # Complete
        await heart.subtasks.complete(subtask.id, "Done!")
        final = await heart.subtasks.get(subtask.id)
        assert final.status == "completed"
        assert final.result == "Done!"

        # Verify counts
        counts = await heart.subtasks.count_by_status()
        assert counts.get("completed", 0) >= 1

        await heart.close()

    async def test_full_schedule_lifecycle(self, db, settings):
        """Create recurring → fire → advance → verify."""
        heart = Heart(db, settings)

        schedule = await heart.schedules.create(
            task="Recurring integration test",
            schedule_type="recurring",
            interval_seconds=3600,
            max_fires=2,
        )

        # Simulate firing twice
        now = datetime.now(UTC)
        await heart.schedules.advance(schedule.id, now)
        s1 = await heart.schedules.get(schedule.id)
        assert s1.fire_count == 1
        assert s1.active is True

        await heart.schedules.advance(schedule.id, now + timedelta(hours=1))
        s2 = await heart.schedules.get(schedule.id)
        assert s2.fire_count == 2
        assert s2.active is False  # max_fires reached

        await heart.close()
```

**Step 2: Run all tests**

```bash
uv run pytest tests/test_subtasks.py tests/test_schedules.py tests/test_time_parser.py -v
```
Expected: All tests PASS.

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --timeout=60 -q
```
Expected: All tests PASS. No regressions.

**Step 4: Commit**

```bash
git add tests/test_subtasks.py
git commit -m "test(011.1): integration tests for subtask and schedule lifecycle"
```

---

## Summary

| Task | Component | Est. Lines |
|------|-----------|-----------|
| 1 | croniter dependency | 0 (config only) |
| 2 | Database migration | ~40 |
| 3 | ORM models | ~60 |
| 4 | SubtaskManager | ~140 |
| 5 | ScheduleManager | ~110 |
| 6 | Heart wiring | ~5 |
| 7 | Time parser | ~90 |
| 8 | Config settings | ~12 |
| 9 | SubtaskWorkerPool | ~160 |
| 10 | TaskScheduler | ~65 |
| 11 | Agent tools | ~200 |
| 12 | REST endpoints | ~150 |
| 13 | Startup wiring | ~30 |
| 14 | Runner prefix support | ~5 |
| 15 | Integration tests | ~60 |

**Total:** ~1125 lines across 15 tasks. 15 commits.
