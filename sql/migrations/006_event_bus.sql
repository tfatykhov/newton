-- 006: Event Bus schema additions
ALTER TABLE heart.episodes ADD COLUMN IF NOT EXISTS structured_summary JSONB;
ALTER TABLE heart.episodes ADD COLUMN IF NOT EXISTS user_id VARCHAR(100);
ALTER TABLE heart.episodes ADD COLUMN IF NOT EXISTS user_display_name VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON heart.episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_summary_gin ON heart.episodes USING GIN(structured_summary);
