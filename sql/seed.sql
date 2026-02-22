-- =============================================================================
-- Nous Seed Data — seed.sql
-- Default agent, 6 cognitive frames, 4 guardrails
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Default agent
-- ---------------------------------------------------------------------------
INSERT INTO nous_system.agents (id, name, description, config) VALUES (
    'nous-default',
    'Nous',
    'A thinking agent that learns from experience',
    '{
        "identity": {"traits": ["analytical", "cautious", "curious"]},
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "working_memory_capacity": 20,
        "auto_extract_facts": true,
        "auto_create_censors": true,
        "censor_escalation": true
    }'::jsonb
);

-- ---------------------------------------------------------------------------
-- Default cognitive frames
-- ---------------------------------------------------------------------------
INSERT INTO nous_system.frames (id, agent_id, name, description, activation_patterns, default_category) VALUES
    ('task', 'nous-default', 'Task Execution', 'Focused on completing a specific task', ARRAY['build', 'fix', 'create', 'implement', 'deploy'], 'tooling'),
    ('question', 'nous-default', 'Question Answering', 'Answering questions, looking things up', ARRAY['what', 'how', 'why', 'explain', 'tell me'], 'process'),
    ('decision', 'nous-default', 'Decision Making', 'Evaluating options, choosing a path', ARRAY['should', 'choose', 'decide', 'compare', 'trade-off'], 'architecture'),
    ('creative', 'nous-default', 'Creative', 'Brainstorming, ideation, exploration', ARRAY['imagine', 'brainstorm', 'what if', 'design', 'explore'], 'architecture'),
    ('conversation', 'nous-default', 'Conversation', 'Casual or social interaction', ARRAY['hello', 'hi', 'thanks', 'how are you'], 'process'),
    ('debug', 'nous-default', 'Debug', 'Investigating problems, tracing errors', ARRAY['error', 'bug', 'broken', 'failing', 'crash', 'wrong'], 'tooling');

-- ---------------------------------------------------------------------------
-- Default guardrails
-- ---------------------------------------------------------------------------
INSERT INTO brain.guardrails (agent_id, name, description, condition, severity) VALUES
    ('nous-default', 'no-high-stakes-low-confidence', 'Block high-stakes decisions with low confidence', '{"stakes": "high", "confidence_lt": 0.5}', 'block'),
    ('nous-default', 'no-critical-without-review', 'Block critical-stakes without explicit review', '{"stakes": "critical"}', 'block'),
    ('nous-default', 'require-reasons', 'Block decisions without at least one reason', '{"reason_count_lt": 1}', 'block'),
    ('nous-default', 'low-quality-recording', 'Block low-quality decisions (missing tags/pattern)', '{"quality_lt": 0.5}', 'block');
