# Design: 011.1 Subtasks & Scheduling

**Date:** 2026-02-28
**Implements:** F009 (Async Subtasks)
**Spec:** `docs/implementation/011.1-subtasks-and-scheduling.md`
**Approach:** Spec-faithful — managers in Heart, workers/scheduler in handlers, tools in api/tools

## Decisions

- **All 3 phases** in one branch (subtasks + scheduling + management + REST)
- **Heart schema** for both tables (subtasks are agent activities)
- **dateutil + croniter** for time parsing (proper cron support)
- **Direct Telegram API** for notifications (httpx POST, no coupling to bot process)
- **Approach A** — follow spec file layout exactly

## Database Schema

Two tables in `heart` schema, migration `sql/migrations/010_subtasks.sql`.

### heart.subtasks

Queue for immediate and scheduled work.

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | gen_random_uuid() |
| agent_id | VARCHAR NOT NULL | Multi-agent scoping |
| parent_session_id | VARCHAR | Spawning conversation or `schedule-{id}` |
| task | TEXT NOT NULL | Instruction for the subtask |
| priority | INTEGER DEFAULT 100 | 50=urgent, 100=normal, 200=low |
| status | VARCHAR DEFAULT 'pending' | pending/running/completed/failed/cancelled |
| result | TEXT | Completion result |
| error | TEXT | Failure reason |
| worker_id | VARCHAR | Which worker claimed it |
| timeout_seconds | INTEGER DEFAULT 120 | Max execution time |
| notify | BOOLEAN DEFAULT TRUE | Send Telegram notification |
| created_at | TIMESTAMPTZ | now() |
| started_at | TIMESTAMPTZ | When worker claimed |
| completed_at | TIMESTAMPTZ | When finished |
| metadata | JSONB DEFAULT '{}' | Extensible |

Index: `(priority, created_at) WHERE status = 'pending'`

Dequeue: `SELECT FOR UPDATE SKIP LOCKED`

### heart.schedules

Timers and recurring jobs.

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | gen_random_uuid() |
| agent_id | VARCHAR NOT NULL | Multi-agent scoping |
| task | TEXT NOT NULL | Instruction to execute |
| schedule_type | VARCHAR NOT NULL | 'once' or 'recurring' |
| fire_at | TIMESTAMPTZ | For one-shot |
| interval_seconds | INTEGER | For recurring |
| cron_expr | VARCHAR | Optional cron expression |
| active | BOOLEAN DEFAULT TRUE | Soft deactivation |
| last_fired_at | TIMESTAMPTZ | Last execution time |
| next_fire_at | TIMESTAMPTZ | For scheduler queries |
| fire_count | INTEGER DEFAULT 0 | Executions so far |
| max_fires | INTEGER | NULL = unlimited |
| notify | BOOLEAN DEFAULT TRUE | Send Telegram notification |
| timeout_seconds | INTEGER DEFAULT 120 | Per-subtask timeout |
| created_at | TIMESTAMPTZ | now() |
| created_by_session | VARCHAR | Which session created it |
| metadata | JSONB DEFAULT '{}' | Extensible |

Index: `(next_fire_at) WHERE active = TRUE`

## Storage Layer

### ORM Models (`nous/storage/models.py`)

`Subtask` and `Schedule` models — `mapped_column()`, agent-scoped, JSONB metadata, `server_default=func.now()`.

### SubtaskManager (`nous/heart/subtasks.py`, ~120 lines)

- `create_subtask()` — insert with priority mapping, enforce max pending (5)
- `dequeue(worker_id)` — `SELECT FOR UPDATE SKIP LOCKED`, set running
- `complete_subtask(id, result)` — set completed
- `fail_subtask(id, error)` — set failed
- `cancel_subtask(id)` — set cancelled (pending only)
- `list_subtasks(status, limit)` — filtered query
- `get_subtask(id)` — single fetch
- `reclaim_stale(timeout)` — re-enqueue stuck tasks (heartbeat recovery)

### ScheduleManager (`nous/heart/schedules.py`, ~100 lines)

- `create_schedule()` — insert, compute next_fire_at
- `get_due_schedules(now)` — WHERE active AND next_fire_at <= now
- `advance_schedule(id, fired_at)` — increment counter, compute next fire, check max_fires
- `deactivate_schedule(id)` — set active=FALSE
- `list_schedules(active_only)` — filtered query
- `get_schedule(id)` — single fetch

### Heart wiring (`nous/heart/heart.py`)

Heart creates SubtaskManager and ScheduleManager in `__init__`, exposes as properties.

## Worker Pool & Scheduler

### SubtaskWorkerPool (`nous/handlers/subtask_worker.py`, ~150 lines)

- Takes: AgentRunner, Heart, Settings, EventBus, httpx.AsyncClient
- `start()` — spawn N worker asyncio tasks (default 2)
- `stop()` — cancel all workers
- Worker loop: poll dequeue → sleep if empty → execute if found
- Execution: `runner.run_turn()` with subtask-scoped session ID and system prompt prefix
- Timeout via `asyncio.wait_for()`
- On success: complete, emit `subtask_completed`, Telegram notify
- On failure: fail, emit `subtask_failed`, Telegram notify with error
- Concurrency guard: check running < max_concurrent (3) before executing
- Stale reclaim on startup

### Telegram Notifications

Direct `httpx.post()` to `api.telegram.org/bot{token}/sendMessage`.
Config: `NOUS_TELEGRAM_BOT_TOKEN`, `NOUS_TELEGRAM_CHAT_ID`.
Format: emoji + task summary + result (truncated 4096 chars).

### TaskScheduler (`nous/handlers/task_scheduler.py`, ~60 lines)

- Takes: Heart, Settings
- `start()` — spawn single asyncio check loop
- `stop()` — cancel task
- Loop: sleep 60s → get_due_schedules → create subtask for each
- One-shot: deactivate after firing
- Recurring: advance (next fire time, increment counter, check max_fires)

## Tools

4 new tools registered in `nous/api/tools.py`:

| Tool | Frames | Description |
|------|--------|-------------|
| `spawn_task` | task, conversation, debug | Spawn background subtask |
| `schedule_task` | task, conversation, debug | Create timer or recurring job |
| `list_tasks` | all | List subtasks and schedules |
| `cancel_task` | all | Cancel subtask or deactivate schedule |

**Recursion prevention:** subtask sessions do NOT get `spawn_task` registered. Depth = 1.

### Time Parsing (`nous/handlers/time_parser.py`, ~80 lines)

`parse_when(when) -> datetime`:
- ISO 8601 via `dateutil.parser.parse()`
- Relative via regex: `r"in\s+(\d+)\s+(minute|hour|day|week)s?"`
- Natural dates via dateutil fallback ("tomorrow 9am", "next monday 8am EST")
- Reject past times

`parse_every(every) -> tuple[int | None, str | None]`:
- Returns (interval_seconds, cron_expr)
- Simple intervals via regex
- Daily/weekly patterns → croniter cron expressions
- croniter validates and computes next_fire_at

## REST Endpoints

Added to `nous/api/rest.py`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/subtasks` | List (status, limit params) |
| GET | `/subtasks/{id}` | Detail + result |
| DELETE | `/subtasks/{id}` | Cancel pending |
| GET | `/schedules` | List (active_only param) |
| POST | `/schedules` | Create externally |
| DELETE | `/schedules/{id}` | Deactivate |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `NOUS_SUBTASK_ENABLED` | true | Master toggle |
| `NOUS_SUBTASK_WORKERS` | 2 | Worker count |
| `NOUS_SUBTASK_POLL_INTERVAL` | 2.0 | Dequeue poll seconds |
| `NOUS_SUBTASK_DEFAULT_TIMEOUT` | 120 | Default timeout |
| `NOUS_SUBTASK_MAX_TIMEOUT` | 600 | Max timeout |
| `NOUS_SUBTASK_MAX_CONCURRENT` | 3 | Max running |
| `NOUS_SCHEDULE_ENABLED` | true | Master toggle |
| `NOUS_SCHEDULE_CHECK_INTERVAL` | 60 | Due-task check seconds |
| `NOUS_TELEGRAM_BOT_TOKEN` | — | For notifications |
| `NOUS_TELEGRAM_CHAT_ID` | — | Target chat |

## Startup Wiring (`nous/main.py`)

1. After AgentRunner: create SubtaskWorkerPool if enabled, `start()`
2. Create TaskScheduler if enabled, `start()`
3. Register 4 tools (skip if disabled)
4. Shutdown: `stop()` workers and scheduler before other components

## Files Changed

| File | Change | Est. Lines |
|------|--------|-----------|
| `sql/migrations/010_subtasks.sql` | Subtasks + schedules tables | ~40 |
| `nous/storage/models.py` | ORM models | ~50 |
| `nous/heart/subtasks.py` | SubtaskManager | ~120 |
| `nous/heart/schedules.py` | ScheduleManager | ~100 |
| `nous/heart/heart.py` | Wire managers | ~20 |
| `nous/handlers/subtask_worker.py` | Worker pool | ~150 |
| `nous/handlers/task_scheduler.py` | Scheduler loop | ~60 |
| `nous/handlers/time_parser.py` | Time parsing | ~80 |
| `nous/api/tools.py` | 4 new tools | ~80 |
| `nous/api/rest.py` | 6 REST endpoints | ~60 |
| `nous/config.py` | New settings | ~15 |
| `nous/main.py` | Startup wiring | ~20 |
| `tests/test_subtasks.py` | Queue, worker, timeout tests | ~120 |
| `tests/test_schedules.py` | Schedule, firing, recurring tests | ~100 |

**Total:** ~1015 lines. 2 tables, 4 tools, 6 endpoints.
