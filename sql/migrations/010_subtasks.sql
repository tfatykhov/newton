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

-- Result delivery tracking
ALTER TABLE heart.subtasks ADD COLUMN IF NOT EXISTS delivered BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_subtasks_undelivered
    ON heart.subtasks (parent_session_id, created_at)
    WHERE status IN ('completed', 'failed') AND delivered = FALSE;
