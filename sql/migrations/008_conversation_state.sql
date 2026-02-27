-- 008: Conversation state table for compaction (spec 008.1 Phase 3)
-- Stores conversation messages + summary between compaction cycles

CREATE TABLE IF NOT EXISTS heart.conversation_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT,
    messages JSONB,
    turn_count INT NOT NULL DEFAULT 0,
    compaction_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(agent_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_state_agent
    ON heart.conversation_state(agent_id);
DROP TRIGGER IF EXISTS set_updated_at ON heart.conversation_state;
CREATE TRIGGER set_updated_at BEFORE UPDATE ON heart.conversation_state
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();
