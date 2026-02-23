# F001: Brain Module (Decision Intelligence)

**Status:** Shipped
**Priority:** P0 — Core organ, everything depends on this
**Origin:** Cognition Engines principles, reimplemented as embedded library

## Summary

Nous's Brain is the decision intelligence organ. It handles decision recording, deliberation traces, confidence calibration, guardrails, and decision relationships. Built as a Python module that runs in-process — no MCP, no HTTP, no external service.

Embodies the proven principles from Cognition Engines but designed from the ground up for embedded use inside Nous.

## Capabilities

### Decision Recording
- Record decisions with confidence, category, stakes, tags, reasoning
- Bridge definitions (structure + function) for dual-angle recall
- Pattern field for conceptual-level description
- Quality scoring — reject decisions missing tags/pattern/reasons

### Deliberation Traces
- Micro-thoughts captured during reasoning
- Attached to decisions, ordered by timestamp
- "Frozen reflection" — capture mental state before work transforms it

### Guardrails (Censors)
- Rule-based checks before actions
- Configurable severity: warn → block → absolute
- Escalation based on repeated triggers

### Calibration
- Brier score tracking (predicted confidence vs actual outcomes)
- Per-category accuracy
- Per-reason-type effectiveness
- Drift detection (rolling window accuracy)

### Decision Graph
- Relationships between decisions (supports, contradicts, supersedes, related_to)
- Auto-linking based on semantic similarity
- Neighbor traversal for context exploration

### Semantic Search
- Hybrid search: vector similarity + keyword matching
- Bridge-side search: find by function ("what solved this?") or structure ("where did we use this?")

## Interface

```python
from nous.brain import Brain

brain = Brain(db_pool)

# Record
decision = await brain.record(
    description="Chose Postgres for unified storage",
    confidence=0.85,
    category="architecture",
    stakes="medium",
    tags=["postgres", "storage", "infrastructure"],
    pattern="Unify infrastructure to reduce operational surface",
    reasons=[
        {"type": "analysis", "text": "One DB simpler than three"},
        {"type": "empirical", "text": "CE SQLite migration proved consolidation works"}
    ]
)

# Think
await brain.think(decision.id, "Considered Qdrant but unification wins over raw speed")

# Query
similar = await brain.query("database storage decision", limit=5)

# Check guardrails
result = await brain.check_guardrails(
    description="Delete production database",
    stakes="critical",
    confidence=0.9
)
# result.blocked == True

# Review outcome
await brain.review(decision.id, outcome="success", result="Postgres performing well")

# Calibration
cal = await brain.calibration()
# cal.brier_score, cal.accuracy, cal.total_decisions
```

## Data Model

Stored in `brain` schema in shared Postgres:

```sql
CREATE TABLE brain.decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    category VARCHAR(50),
    stakes VARCHAR(20),
    context TEXT,
    pattern TEXT,
    quality_score FLOAT,
    outcome VARCHAR(20),           -- pending, success, partial, failure
    outcome_result TEXT,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);

CREATE TABLE brain.tags (
    decision_id UUID REFERENCES brain.decisions(id),
    tag VARCHAR(100),
    PRIMARY KEY (decision_id, tag)
);

CREATE TABLE brain.reasons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES brain.decisions(id),
    type VARCHAR(50),              -- analysis, pattern, empirical, authority, intuition, analogy, elimination, constraint
    text TEXT NOT NULL
);

CREATE TABLE brain.bridge (
    decision_id UUID PRIMARY KEY REFERENCES brain.decisions(id),
    structure TEXT,                 -- What it looks like
    function TEXT                   -- What it does
);

CREATE TABLE brain.deliberation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES brain.decisions(id),
    agent_id VARCHAR(100),
    thought TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE brain.graph_edges (
    source_id UUID REFERENCES brain.decisions(id),
    target_id UUID REFERENCES brain.decisions(id),
    relation VARCHAR(50),          -- supports, contradicts, supersedes, related_to
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_id, target_id, relation)
);

CREATE TABLE brain.guardrails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    condition JSONB NOT NULL,       -- Rule definition
    severity VARCHAR(20) DEFAULT 'warn',
    active BOOLEAN DEFAULT TRUE,
    activation_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## What We Take from CE

| CE Feature | Nous Brain | Changes |
|-----------|-------------|---------|
| Decision recording | ✅ Same principles | UUID keys, embedded API |
| Deliberation traces | ✅ Same | Simplified tracker (no transport-layer keys) |
| Guardrails | ✅ Same + censor integration | Merged with Heart's censor system |
| Calibration | ✅ Same math | Computed from Postgres directly |
| Bridge definitions | ✅ Same | Auto-extracted same way |
| Quality scoring | ✅ Same | Block low-quality by default |
| Hybrid search | ✅ Same concept | pgvector + tsvector instead of ChromaDB |
| Decision graph | ✅ Same | Postgres foreign keys, not JSONL file |
| Pre-action protocol | ✅ Evolved | Part of cognitive layer hooks, not MCP tool |
| MCP interface | ❌ External only | Internal = Python API |

## Relationship to Cognition Engines

**Same ideas, not same code.**

Cognition Engines (CE) is the research prototype that proved decision intelligence works for AI agents. Nous's Brain applies the same principles - decisions, deliberation, calibration, guardrails, bridge definitions, quality scoring - but is a completely independent implementation designed for embedded use.

| | Cognition Engines | Nous Brain |
|--|---|---|
| **Role** | Standalone decision intelligence server | Embedded organ inside Nous |
| **Interface** | JSON-RPC, MCP, HTTP | Python function calls |
| **Storage** | SQLite + ChromaDB | PostgreSQL + pgvector |
| **Transport** | Network (MCP/HTTP) | In-process (zero overhead) |
| **Search** | ChromaDB vectors + SQLite FTS | pgvector + tsvector (single query) |
| **IDs** | String-based | UUIDs |
| **Audience** | Any AI agent needing decision memory | Nous agents specifically |
| **Codebase** | Independent | Independent |

CE continues to exist as a standalone tool. Nous's Brain is purpose-built for embedding. Both evolve independently. The shared asset is the *philosophy*, not the code.

---

*"Consciousness is menu lists, not deep access."* — Minsky Ch 6
