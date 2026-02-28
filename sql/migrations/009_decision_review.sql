-- 009_decision_review.sql
-- Add session_id and reviewer columns to brain.decisions
-- Part of spec 008.5: Decision Review Loop

ALTER TABLE brain.decisions ADD COLUMN IF NOT EXISTS session_id VARCHAR(100);
ALTER TABLE brain.decisions ADD COLUMN IF NOT EXISTS reviewer VARCHAR(50);

CREATE INDEX IF NOT EXISTS idx_decisions_session_id ON brain.decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_decisions_unreviewed ON brain.decisions(reviewed_at) WHERE reviewed_at IS NULL;
