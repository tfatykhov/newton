# 002: Brain Module — Decision Intelligence Organ

**Status:** Shipped (PR #2)
**Priority:** P0 — Core organ, everything depends on this
**Estimated Effort:** 8-12 hours
**Prerequisites:** 001-postgres-scaffold (merged)
**Feature Spec:** [F001-brain-module.md](../features/F001-brain-module.md)

## Objective

Implement the Brain — Nous's decision intelligence organ. An async Python module that records decisions, captures deliberation traces, enforces guardrails, computes calibration metrics, manages decision relationships, and performs hybrid search. All operations are in-process Python calls against the shared Postgres connection pool. No HTTP, no MCP — pure embedded library.

After this phase, any Python code can `from nous.brain import Brain` and have full decision intelligence.

## Architecture

```
Brain (nous/brain/brain.py)
├── record()        → Decision with tags, reasons, bridge, quality score
├── update()        → Modify decision description, context, pattern
├── think()         → Attach deliberation thought to a decision
├── get()           → Fetch single decision with all relations
├── query()         → Hybrid search (vector + keyword)
├── check()         → Evaluate guardrails before action
├── review()        → Record outcome (success/partial/failure)
├── calibration()   → Compute Brier score, accuracy, breakdowns
├── link()          → Create graph edge between decisions
├── neighbors()     → Get connected decisions from graph
├── auto_link()     → Find and link similar decisions automatically
└── emit_event()    → Log cognitive events to nous_system.events
         │
         ├── EmbeddingProvider (nous/brain/embeddings.py)
         │   └── OpenAI text-embedding-3-small
         │
         ├── QualityScorer (nous/brain/quality.py)
         │   └── Score based on tags, reasons, pattern presence
         │
         ├── GuardrailEngine (nous/brain/guardrails.py)
         │   └── Evaluate JSONB conditions against action context
         │
         ├── CalibrationEngine (nous/brain/calibration.py)
         │   └── Brier score, accuracy, per-category, per-reason-type
         │
         └── BridgeExtractor (nous/brain/bridge.py)
             └── Auto-extract structure + function descriptions
```

## Deliverables

### File Structure

```
nous/brain/
├── __init__.py          # Public exports: Brain, BrainConfig
├── brain.py             # Main Brain class — public API
├── embeddings.py        # EmbeddingProvider (OpenAI API)
├── quality.py           # Quality scoring logic
├── guardrails.py        # Guardrail evaluation engine
├── calibration.py       # Calibration math (Brier, accuracy, breakdowns)
├── bridge.py            # Bridge extraction (structure + function)
└── schemas.py           # Pydantic models for Brain inputs/outputs

tests/
├── test_brain.py        # Integration tests for Brain public API
├── test_guardrails.py   # Unit tests for guardrail evaluation
├── test_calibration.py  # Unit tests for calibration math
└── test_quality.py      # Unit tests for quality scoring
```

---

### 1. `nous/brain/schemas.py` — Data Transfer Objects

Pydantic models for all Brain inputs and outputs. These are the public contract.

```python
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime


class ReasonInput(BaseModel):
    type: str  # analysis, pattern, empirical, authority, intuition, analogy, elimination, constraint
    text: str


class RecordInput(BaseModel):
    """Input for brain.record()"""
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: str  # architecture, process, tooling, security, integration
    stakes: str  # low, medium, high, critical
    context: str | None = None
    pattern: str | None = None
    tags: list[str] = []
    reasons: list[ReasonInput] = []


class DecisionSummary(BaseModel):
    """Lightweight decision returned from queries."""
    id: UUID
    description: str
    confidence: float
    category: str
    stakes: str
    outcome: str
    pattern: str | None
    tags: list[str]
    score: float | None = None  # Relevance score from search
    created_at: datetime


class DecisionDetail(BaseModel):
    """Full decision with all relations."""
    id: UUID
    agent_id: str
    description: str
    context: str | None
    pattern: str | None
    confidence: float
    category: str
    stakes: str
    quality_score: float | None
    outcome: str
    outcome_result: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    tags: list[str]
    reasons: list[ReasonInput]
    bridge: BridgeInfo | None
    thoughts: list[ThoughtInfo]


class BridgeInfo(BaseModel):
    structure: str | None
    function: str | None


class ThoughtInfo(BaseModel):
    id: UUID
    text: str
    created_at: datetime


class GuardrailResult(BaseModel):
    """Result of guardrail check."""
    allowed: bool
    blocked_by: list[str] = []  # Names of blocking guardrails
    warnings: list[str] = []    # Names of warning guardrails


class CalibrationReport(BaseModel):
    """Calibration metrics."""
    total_decisions: int
    reviewed_decisions: int
    brier_score: float | None
    accuracy: float | None
    confidence_mean: float | None
    confidence_stddev: float | None
    category_stats: dict  # {category: {count, accuracy, brier}}
    reason_type_stats: dict  # {type: {count, brier}}


class GraphEdgeInfo(BaseModel):
    source_id: UUID
    target_id: UUID
    relation: str
    weight: float
    auto_linked: bool
```

---

### 2. `nous/brain/embeddings.py` — Embedding Provider

```python
"""Generate embeddings via OpenAI API."""

import httpx


class EmbeddingProvider:
    """Async embedding generation using OpenAI text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.model = model
        self.dimensions = dimensions
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self._client.post(
            "/embeddings",
            json={"model": self.model, "input": text, "dimensions": self.dimensions},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (single API call)."""
        response = await self._client.post(
            "/embeddings",
            json={"model": self.model, "input": texts, "dimensions": self.dimensions},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    async def close(self) -> None:
        await self._client.aclose()
```

**Implementation notes:**
- Use `httpx.AsyncClient` for connection pooling (module-level client, not per-request)
- Batch endpoint for bulk operations (up to 2048 texts per call)
- Handle rate limits with exponential backoff (optional, P2 enhancement)
- If `api_key` is empty/None, skip embedding generation (return None). This allows development without an API key — queries without embeddings fall back to keyword-only search.

---

### 3. `nous/brain/quality.py` — Quality Scoring

Compute a quality score (0.0-1.0) for decisions based on metadata completeness.

```python
def compute_quality_score(
    tags: list[str],
    reasons: list[dict],
    pattern: str | None,
    context: str | None,
) -> float:
    """
    Score decision quality based on metadata completeness.
    
    Scoring:
    - Has tags (≥1):        +0.25
    - Has reasons (≥1):     +0.25
    - Has pattern:          +0.25
    - Has context:          +0.15
    - Reason diversity (≥2 types): +0.10
    
    Total possible: 1.0
    Block threshold: < 0.5 (enforced by guardrail, not here)
    """
```

**Notes:**
- Pure function, no DB access, easily testable
- Matches CE's quality scoring logic
- Reason diversity = number of unique reason types, not count

---

### 4. `nous/brain/guardrails.py` — Guardrail Engine

Evaluate guardrail conditions against a proposed action.

```python
class GuardrailEngine:
    """Evaluate JSONB guardrail conditions against action context."""

    async def check(
        self, session: AsyncSession, agent_id: str,
        description: str, stakes: str, confidence: float,
        tags: list[str] | None = None,
        reasons: list[dict] | None = None,
        pattern: str | None = None,
    ) -> GuardrailResult:
        """
        Check all active guardrails for the agent.
        Returns GuardrailResult with allowed/blocked_by/warnings.
        
        Condition matching rules (JSONB):
        - "stakes": "high"          → exact match on stakes
        - "confidence_lt": 0.5      → confidence < value
        - "reason_count_lt": 1      → len(reasons) < value  
        - "quality_lt": 0.5         → quality_score < value
        
        After evaluation:
        - Increment activation_count for triggered guardrails
        - Update last_activated timestamp
        """
```

**Implementation notes:**
- Load all active guardrails for agent_id from `brain.guardrails`
- Evaluate each condition against the action context
- `block` and `absolute` severity → `allowed = False`
- `warn` severity → `allowed = True`, added to warnings list
- `absolute` guardrails cannot be overridden (future: override mechanism for block)
- Log guardrail triggers to `nous_system.events` with event_type `guardrail_blocked` or `guardrail_warned`

---

### 5. `nous/brain/calibration.py` — Calibration Engine

Compute calibration metrics from reviewed decisions.

```python
class CalibrationEngine:
    """Compute calibration metrics from Postgres data."""

    async def compute(self, session: AsyncSession, agent_id: str) -> CalibrationReport:
        """
        Compute full calibration report.
        
        Metrics:
        1. Brier Score = mean((confidence - outcome_binary)²)
           - outcome_binary: success=1, partial=0.5, failure=0
        
        2. Accuracy = reviewed decisions with correct outcome / total reviewed
           - "correct" = confidence >= 0.5 AND outcome in (success, partial)
                      OR confidence < 0.5 AND outcome = failure
        
        3. Confidence stats = mean and stddev of all confidence values
        
        4. Per-category breakdown = {category: {count, accuracy, brier_score}}
        
        5. Per-reason-type breakdown = {type: {count, brier_score}}
           - Computed by joining decisions with their reasons
        
        All computed via SQL aggregation, not loading all rows into Python.
        """
```

**Implementation notes:**
- Use raw SQL or SQLAlchemy `func` for aggregations — do NOT load all decisions into memory
- Brier score formula: `AVG(POWER(confidence - CASE outcome WHEN 'success' THEN 1.0 WHEN 'partial' THEN 0.5 ELSE 0.0 END, 2))`
- Per-category: GROUP BY category
- Per-reason-type: JOIN with `brain.decision_reasons`, GROUP BY type
- Only include decisions where `outcome != 'pending'`
- If no reviewed decisions, return nulls for brier/accuracy

---

### 6. `nous/brain/bridge.py` — Bridge Extractor

Auto-extract structure and function descriptions from decision text.

```python
class BridgeExtractor:
    """Extract bridge definitions (structure + function) from decision text."""

    def extract(self, description: str, context: str | None, pattern: str | None) -> BridgeInfo:
        """
        Extract structure (what it looks like) and function (what it does)
        from decision text using heuristic rules.
        
        Heuristics:
        - Structure: extracted from description — concrete nouns, tools, names
          e.g., "PostgreSQL schema with 3 schemas and 18 tables"
        - Function: extracted from pattern or context — verbs, purposes
          e.g., "Provides persistent, searchable memory for an AI agent"
        
        If pattern is set, use it as function.
        First sentence of description becomes structure.
        """
```

**Implementation notes:**
- Heuristic-based for now (no LLM calls). Keep it simple:
  - `structure` = first sentence of description (truncated to 200 chars)
  - `function` = pattern if available, else first sentence of context
- This can be upgraded to LLM-based extraction later (F002+)
- The bridge is stored in `brain.decision_bridge` table

---

### 7. `nous/brain/brain.py` — Main Brain Class

The public API. All methods are async and take/return Pydantic models.

```python
class Brain:
    """Decision intelligence organ for Nous agents."""

    def __init__(self, database: Database, settings: Settings, embedding_provider: EmbeddingProvider | None = None):
        self.db = database
        self.settings = settings
        self.embeddings = embedding_provider
        self.quality = QualityScorer()
        self.guardrails = GuardrailEngine()
        self.calibration = CalibrationEngine()
        self.bridge_extractor = BridgeExtractor()
        self.agent_id = settings.agent_id
```

#### Methods:

**`async def record(self, input: RecordInput) -> DecisionDetail`**
1. Compute quality score via `quality.compute_quality_score()`
2. Generate embedding for `f"{input.description} {input.context or ''} {input.pattern or ''}"` (skip if no embedding provider)
3. Extract bridge via `bridge_extractor.extract()`
4. Insert into `brain.decisions` with all fields
5. Insert tags into `brain.decision_tags`
6. Insert reasons into `brain.decision_reasons`
7. Insert bridge into `brain.decision_bridge`
8. Call `auto_link()` to find and link similar decisions
9. Emit event `decision_recorded`
10. Return full `DecisionDetail`

**`async def update(self, decision_id: UUID, description: str | None = None, context: str | None = None, pattern: str | None = None) -> DecisionDetail`**
1. Fetch existing decision
2. Update provided fields
3. Re-compute quality score if tags/reasons changed
4. Re-generate embedding if description/context/pattern changed
5. Re-extract bridge if text changed
6. Emit event `decision_updated`
7. Return updated `DecisionDetail`

**`async def think(self, decision_id: UUID, text: str) -> ThoughtInfo`**
1. Insert into `brain.thoughts` with decision_id, agent_id, text
2. Return `ThoughtInfo`

**`async def get(self, decision_id: UUID) -> DecisionDetail | None`**
1. Fetch decision with eager-loaded tags, reasons, bridge, thoughts
2. Return `DecisionDetail` or None if not found

**`async def query(self, query_text: str, limit: int = 10, category: str | None = None, stakes: str | None = None, outcome: str | None = None, bridge_side: str | None = None) -> list[DecisionSummary]`**
1. Generate embedding for query_text (if embedding provider available)
2. Build hybrid search query:
   ```sql
   WITH semantic AS (
       SELECT id, 1 - (embedding <=> :query_embedding) AS score
       FROM brain.decisions
       WHERE agent_id = :agent_id AND embedding IS NOT NULL
       ORDER BY embedding <=> :query_embedding
       LIMIT :limit * 3
   ),
   keyword AS (
       SELECT id, ts_rank_cd(search_tsv, plainto_tsquery('english', :query_text)) AS score
       FROM brain.decisions
       WHERE agent_id = :agent_id
         AND search_tsv @@ plainto_tsquery('english', :query_text)
       LIMIT :limit * 3
   )
   SELECT COALESCE(s.id, k.id) AS id,
       (COALESCE(s.score, 0) * 0.7 + COALESCE(k.score, 0) * 0.3) AS combined_score
   FROM semantic s
   FULL OUTER JOIN keyword k ON s.id = k.id
   ORDER BY combined_score DESC
   LIMIT :limit
   ```
3. If `bridge_side = "function"` → search against `brain.decision_bridge.function` column
4. If `bridge_side = "structure"` → search against `brain.decision_bridge.structure` column
5. Apply optional filters (category, stakes, outcome) as WHERE clauses
6. If no embedding provider, fall back to keyword-only search
7. Join with `brain.decision_tags` to populate tags in results
8. Return list of `DecisionSummary`

**`async def check(self, description: str, stakes: str, confidence: float, tags: list[str] | None = None, reasons: list[dict] | None = None, pattern: str | None = None) -> GuardrailResult`**
1. Delegate to `guardrails.check()` with all parameters
2. Return `GuardrailResult`

**`async def review(self, decision_id: UUID, outcome: str, result: str | None = None) -> DecisionDetail`**
1. Update `brain.decisions` set outcome, outcome_result, reviewed_at = now()
2. Emit event `decision_reviewed`
3. Return updated `DecisionDetail`

**`async def get_calibration(self) -> CalibrationReport`**
1. Delegate to `calibration.compute()`
2. Return `CalibrationReport`

**`async def link(self, source_id: UUID, target_id: UUID, relation: str, weight: float = 1.0) -> GraphEdgeInfo`**
1. Insert into `brain.graph_edges` (auto_linked=False)
2. Emit event `decisions_linked`
3. Return `GraphEdgeInfo`

**`async def neighbors(self, decision_id: UUID, relation: str | None = None, limit: int = 10) -> list[DecisionSummary]`**
1. Query `brain.graph_edges` where source_id OR target_id = decision_id
2. Optionally filter by relation type
3. Fetch connected decisions
4. Return list of `DecisionSummary`

**`async def auto_link(self, decision_id: UUID, threshold: float = 0.85, max_links: int = 3) -> list[GraphEdgeInfo]`**
1. Get the decision's embedding
2. Find top N similar decisions by cosine similarity (above threshold)
3. Create `related_to` edges with auto_linked=True, weight=cosine_similarity
4. Skip if decision already linked or is self
5. Return list of created `GraphEdgeInfo`

**`async def emit_event(self, event_type: str, data: dict) -> None`**
1. Insert into `nous_system.events` with agent_id, event_type, data
2. Fire-and-forget (don't await in the caller's critical path — use background or just insert)

---

### 8. Configuration

Add to `nous/config.py`:

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Brain settings
    openai_api_key: str = ""  # Empty = no embeddings, keyword-only search
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    auto_link_threshold: float = 0.85
    auto_link_max: int = 3
    quality_block_threshold: float = 0.5
```

---

### 9. Tests

#### `tests/test_brain.py` — Integration Tests

All tests use real Postgres via the SAVEPOINT fixture from 001.

1. **`test_record_decision`** — Record with all fields, verify stored correctly
2. **`test_record_with_tags_and_reasons`** — Verify tags and reasons cascade-inserted
3. **`test_record_computes_quality_score`** — Quality score > 0.5 with tags+reasons+pattern
4. **`test_record_generates_bridge`** — Bridge auto-extracted from description/pattern
5. **`test_record_auto_links`** — Record two similar decisions, verify graph edge created (requires embedding provider mock)
6. **`test_think`** — Attach thought to decision, verify retrieval
7. **`test_get_decision`** — Fetch with all relations populated
8. **`test_get_nonexistent`** — Returns None
9. **`test_update_decision`** — Update description, verify re-scored
10. **`test_query_keyword_only`** — Search without embeddings (keyword fallback)
11. **`test_query_hybrid`** — Search with mock embeddings (both vector + keyword)
12. **`test_query_with_filters`** — Filter by category, stakes, outcome
13. **`test_check_guardrails_allowed`** — Low stakes, high confidence → allowed
14. **`test_check_guardrails_blocked`** — High stakes, low confidence → blocked by seed guardrail
15. **`test_review_decision`** — Set outcome, verify reviewed_at set
16. **`test_calibration_report`** — Record 5 decisions, review 3, verify Brier score
17. **`test_link_decisions`** — Manual link, verify edge created
18. **`test_neighbors`** — Link 3 decisions, query neighbors of middle one
19. **`test_emit_event`** — Verify event written to nous_system.events

#### `tests/test_guardrails.py` — Unit Tests

1. **`test_stakes_match`** — Condition `{"stakes": "high"}` matches high stakes
2. **`test_confidence_lt`** — Condition `{"confidence_lt": 0.5}` matches 0.3
3. **`test_reason_count_lt`** — Condition `{"reason_count_lt": 1}` matches empty reasons
4. **`test_quality_lt`** — Condition `{"quality_lt": 0.5}` matches low quality
5. **`test_multiple_conditions`** — Both conditions must match (AND logic)
6. **`test_warn_vs_block`** — Warn allows, block denies
7. **`test_inactive_guardrail_skipped`** — Disabled guardrails don't trigger
8. **`test_activation_count_incremented`** — Counter goes up on trigger

#### `tests/test_calibration.py` — Unit Tests

1. **`test_brier_perfect`** — All confidence=1.0 with outcome=success → Brier=0.0
2. **`test_brier_worst`** — All confidence=1.0 with outcome=failure → Brier=1.0
3. **`test_brier_partial`** — Mix of outcomes → expected Brier value
4. **`test_accuracy_calculation`** — 3 correct, 1 wrong → 75%
5. **`test_no_reviewed_decisions`** — Returns None for brier/accuracy
6. **`test_per_category_breakdown`** — Stats grouped by category
7. **`test_per_reason_type`** — Stats grouped by reason type

#### `tests/test_quality.py` — Unit Tests

1. **`test_full_quality`** — All fields → score = 1.0
2. **`test_no_metadata`** — No tags, reasons, pattern, context → score = 0.0
3. **`test_tags_only`** — Score = 0.25
4. **`test_reason_diversity_bonus`** — 2+ types → extra 0.10
5. **`test_quality_threshold`** — Below 0.5 is "low quality"

---

## Embedding Provider Mock for Tests

Create a test fixture that returns deterministic embeddings:

```python
# tests/conftest.py (add to existing)

class MockEmbeddingProvider:
    """Returns deterministic embeddings based on text hash."""
    
    async def embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Generate 1536 floats from hash, normalized
        embedding = []
        for i in range(0, 1536):
            byte_val = h[i % 32]
            embedding.append((byte_val / 255.0) * 2 - 1)  # [-1, 1]
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
    
    async def close(self) -> None:
        pass

@pytest.fixture
def mock_embeddings():
    return MockEmbeddingProvider()
```

This avoids hitting OpenAI during tests while still exercising vector storage and search paths.

---

## Acceptance Criteria

1. `from nous.brain import Brain` works
2. `brain.record()` stores decision with tags, reasons, bridge, quality score, embedding
3. `brain.think()` attaches thoughts to decisions
4. `brain.get()` returns full decision with all relations
5. `brain.query()` performs hybrid search (vector + keyword) with filters
6. `brain.query()` falls back to keyword-only when no embedding provider
7. `brain.check()` evaluates all 4 seed guardrails correctly
8. `brain.review()` sets outcome and reviewed_at
9. `brain.get_calibration()` returns correct Brier score, accuracy, breakdowns
10. `brain.link()` creates graph edges between decisions
11. `brain.neighbors()` returns connected decisions
12. `brain.auto_link()` finds and links similar decisions above threshold
13. Quality scoring matches CE behavior (tags+reasons+pattern+context+diversity)
14. All event types logged to `nous_system.events`
15. All tests pass: `pytest tests/test_brain.py tests/test_guardrails.py tests/test_calibration.py tests/test_quality.py -v`

## Non-Goals (This Phase)

- No REST API endpoints (implementation 003+)
- No MCP tools (implementation 003+)
- No LLM-based bridge extraction (heuristic only for now)
- No drift detection (calibration snapshots + trend analysis is future)
- No censor integration with Heart (Brain guardrails only for now)
- No pre-action protocol (that's Cognitive Layer, F003)
- No embedding rate limiting / retry logic (P2 enhancement)

## References

- `docs/features/F001-brain-module.md` — High-level feature spec
- `docs/research/008-database-design.md` — Complete SQL for brain schema
- `nous/storage/models.py` — ORM models (already implemented in 001)
- `sql/init.sql` — Actual table definitions with CHECK constraints
- CE source code — conceptual reference (same ideas, not same code)

---

*"Consciousness is menu lists, not deep access." — Minsky Ch 6*
