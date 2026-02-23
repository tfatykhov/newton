# F002: Heart Module (Memory System)

**Status:** Shipped
**Priority:** P0 — Core organ
**Origin:** New for Nous, informed by Minsky's memory theory
**Detail:** See [008-database-design](../research/008-database-design.md), [009-context-management](../research/009-context-management.md), [010-summarization-strategy](../research/010-summarization-strategy.md)

## Summary

Nous's Heart is the memory organ. It manages episodic, semantic, procedural, working, and censor memory. Like the Brain, it's an embedded Python module — direct function calls, shared Postgres, zero overhead.

Key design principles:
- **Three detail levels** per memory: micro (20 tok), summary (50-100 tok), full (on demand)
- **Frame-adaptive context budgets**: 3K-12K tokens depending on task complexity
- **Automatic lifecycle management**: facts confirm/supersede, episodes trim/archive, censors escalate/retire
- **Event-driven extraction**: episodes auto-produce facts and censor candidates on close

## Capabilities

### Episodic Memory
- Records what happened during interactions
- Auto-generated summaries at session/task end
- Links to Brain decisions made during the episode
- Tracks participants, duration, outcome, surprise level

### Semantic Memory (Facts)
- Stores learned information with provenance
- Lifecycle: learned → confirmed → superseded → deactivated
- Deduplication (don't store the same fact twice)
- Confidence levels (facts can be uncertain)

### Procedural Memory (K-Lines)
- How-to knowledge with Minsky level-bands
- Upper fringe: goals (weakly attached)
- Core: patterns, tools, concepts (strongly attached)
- Lower fringe: implementation details (easily displaced)
- Effectiveness tracking (success/failure per activation)

### Working Memory
- Current session state
- Capacity-limited (evicts low-relevance items)
- Persists across turns within a session
- Cleared on session end (extracted to long-term stores)

### Censor Registry
- Things NOT to do
- Created from failures (auto or manual)
- Severity escalation: warn → block → absolute
- False positive tracking
- Shared with Brain's guardrail system

## Interface

```python
from nous.heart import Heart

heart = Heart(db_pool)

# --- Episodic ---
episode = await heart.record_episode(
    title="Nous architecture discussion with Tim",
    summary="Decided on Postgres + pgvector, embedded Brain/Heart modules",
    participants=["Tim"],
    decision_ids=["abc123", "def456"],
    outcome="success"
)

# --- Semantic ---
await heart.learn(
    "PostgreSQL pgvector uses ivfflat for ANN indexing",
    category="technical",
    subject="pgvector",
    source="Nous architecture research"
)

# Check before storing duplicates
existing = await heart.recall("pgvector indexing", type="fact", limit=1)

# --- Procedural ---
await heart.store_procedure(
    name="Architecture Decision",
    domain="architecture",
    goals=["Make informed technical choices"],
    core_patterns=["Query similar past decisions", "Evaluate at least 2 options", "Record with confidence"],
    core_tools=["brain.query", "brain.record", "brain.think"],
    core_concepts=["Parallel bundles", "Bridge definitions"],
    implementation_notes=["Use brain.check_guardrails before committing"]
)

# Activate (loads into working memory, updates count)
procedure = await heart.activate_procedure("architecture-decision")

# --- Working Memory ---
await heart.focus("Evaluating Postgres vs Qdrant for Nous storage")
await heart.load_context(procedure_id=proc.id)
await heart.load_context(fact_ids=[fact1.id, fact2.id])
context = await heart.get_working_memory()  # Returns all loaded items

# --- Censors ---
await heart.add_censor(
    trigger="pushing directly to main branch",
    reason="Always use feature branches + PR",
    severity="block",
    learned_from_decision="abc123"
)

active_censors = await heart.check_censors("git push origin main")
# Returns [Censor(action="block", reason="Always use feature branches...")]

# --- Unified Recall ---
results = await heart.recall("SQLite migration", limit=10)
# Returns mixed results: episodes + facts + procedures + censors
# Ranked by relevance
```

## Data Model

Stored in `heart` schema in shared Postgres:

```sql
-- Episodic Memory
CREATE TABLE heart.episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    title VARCHAR(500),
    summary TEXT NOT NULL,
    detail TEXT,
    embedding vector(1536),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds INT,
    decision_ids TEXT[],
    participants TEXT[],
    tags TEXT[],
    frame_used VARCHAR(100),
    procedures_activated UUID[],
    outcome VARCHAR(50),
    surprise_level FLOAT,
    lessons_learned TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Semantic Memory
CREATE TABLE heart.facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    category VARCHAR(100),
    subject VARCHAR(500),
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(500),
    source_episode_id UUID REFERENCES heart.episodes(id),
    source_decision_id TEXT,
    learned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_confirmed TIMESTAMPTZ,
    superseded_by UUID,
    active BOOLEAN DEFAULT TRUE,
    tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Procedural Memory (K-Lines)
CREATE TABLE heart.procedures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    name VARCHAR(500) NOT NULL,
    domain VARCHAR(100),
    embedding vector(1536),
    goals TEXT[],
    core_patterns TEXT[],
    core_tools TEXT[],
    core_concepts TEXT[],
    implementation_notes TEXT[],
    activation_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    last_activated TIMESTAMPTZ,
    related_procedures UUID[],
    related_decision_ids TEXT[],
    censors TEXT[],
    tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Censors
CREATE TABLE heart.censors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    trigger_pattern TEXT NOT NULL,
    action VARCHAR(20) NOT NULL DEFAULT 'warn',
    reason TEXT NOT NULL,
    embedding vector(1536),
    learned_from_decision TEXT,
    learned_from_episode UUID REFERENCES heart.episodes(id),
    severity VARCHAR(20) DEFAULT 'warn',
    activation_count INT DEFAULT 0,
    last_activated TIMESTAMPTZ,
    false_positive_count INT DEFAULT 0,
    escalation_threshold INT DEFAULT 3,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Working Memory
CREATE TABLE heart.working_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(100),
    current_task TEXT,
    current_frame VARCHAR(100),
    items JSONB NOT NULL DEFAULT '[]',
    open_threads JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_episodes_embedding ON heart.episodes USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_facts_embedding ON heart.facts USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_procedures_embedding ON heart.procedures USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_censors_embedding ON heart.censors USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_episodes_agent ON heart.episodes(agent_id);
CREATE INDEX idx_facts_agent ON heart.facts(agent_id);
CREATE INDEX idx_facts_active ON heart.facts(active) WHERE active = TRUE;
CREATE INDEX idx_procedures_domain ON heart.procedures(domain);
CREATE INDEX idx_censors_active ON heart.censors(active) WHERE active = TRUE;
```

## Auto-Extraction Pipeline

After each episode, Heart automatically extracts:

```python
async def post_episode_extraction(self, episode: Episode):
    """Extract long-term memories from a completed episode."""
    
    # 1. Extract facts mentioned
    facts = await self.llm.extract_facts(episode.detail)
    for fact in facts:
        existing = await self.recall(fact.content, type="fact", limit=1)
        if existing and existing[0].score > 0.95:
            await self.confirm_fact(existing[0].id)  # Already known
        else:
            await self.learn(fact.content, category=fact.category,
                           source_episode_id=episode.id)
    
    # 2. Update procedure effectiveness
    for proc_id in episode.procedures_activated:
        if episode.outcome == "success":
            await self.record_procedure_success(proc_id)
        elif episode.outcome == "failure":
            await self.record_procedure_failure(proc_id)
    
    # 3. Generate censor candidates from failures
    if episode.outcome == "failure":
        candidates = await self.llm.suggest_censors(episode)
        for candidate in candidates:
            await self.add_censor(
                trigger=candidate.trigger,
                reason=candidate.reason,
                severity="warn",  # Start as warning
                learned_from_episode=episode.id
            )
```

## Brain ↔ Heart Integration

Brain and Heart are separate modules but deeply connected:

| Brain Event | Heart Response |
|------------|----------------|
| Decision recorded | Heart links to current episode |
| Decision failed | Heart creates censor candidate |
| Guardrail triggered | Heart updates censor activation count |
| Calibration drift | Heart flags procedures for review |

| Heart Event | Brain Response |
|------------|----------------|
| Censor activated | Brain logs as guardrail event |
| Procedure loaded | Brain includes in deliberation context |
| New fact learned | Brain updates related decision context |
| Episode completed | Brain reviews pending decisions |

```python
# Both share an event bus for coordination
class NousEvents:
    async def emit(self, event: str, data: dict):
        for handler in self.handlers[event]:
            await handler(data)

events = NousEvents()
brain = Brain(db_pool, events)
heart = Heart(db_pool, events)

# Brain emits "decision_failed" → Heart auto-creates censor
# Heart emits "censor_activated" → Brain logs guardrail event
```

---

*"Memory is reconstruction, not retrieval. You become in those respects more like an earlier version of yourself."* — Minsky Ch 8
