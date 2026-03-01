-- 011.2: Subtask Result Delivery — delivered column for tracking parent injection
ALTER TABLE heart.subtasks ADD COLUMN IF NOT EXISTS delivered BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_subtasks_undelivered
    ON heart.subtasks (parent_session_id, created_at)
    WHERE status IN ('completed', 'failed') AND delivered = FALSE;
