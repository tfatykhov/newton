# 003: Heart Module — Memory System Organ

**Status:** Shipped (PR #3)
**Priority:** P0 — Core organ, Cognitive Layer needs this
**Estimated Effort:** 8-12 hours
**Prerequisites:** 001-postgres-scaffold (merged), 002-brain-module (merged)
**Feature Spec:** [F002-heart-module.md](../features/F002-heart-module.md)

## Objective

Implement the Heart — Nous's memory organ. An async Python module managing five memory types: episodic (what happened), semantic (what we know), procedural (how to do things), censors (what NOT to do), and working memory (current focus). All operations are in-process Python calls against the shared Postgres connection pool, reusing the EmbeddingProvider from Brain.

After this phase, any Python code can `from nous.heart import Heart` and have full memory management.

## Architecture

```
Heart (nous/heart/heart.py)
├── Episodes
│   ├── start_episode()    → Begin tracking an interaction
│   ├── end_episode()      → Close with summary, outcome, lessons
│   ├── get_episode()      → Fetch single episode with relations
│   └── list_episodes()    → Recent episodes for agent
│
├── Facts (Semantic Memory)
│   ├── learn()            → Store new fact with provenance
│   ├── confirm()          → Bump confirmation count + timestamp
│   ├── supersede()        → Mark fact replaced by newer one
│   ├── get_fact()         → Fetch single fact
│   └── search_facts()     → Semantic + keyword search over facts
│
├── Procedures (K-Lines)
│   ├── store_procedure()  → Create procedure with level-bands
│   ├── activate()         → Load into working memory, bump count
│   ├── record_outcome()   → Track success/failure/neutral
│   ├── get_procedure()    → Fetch single procedure
│   └── search_procedures() → Search by domain or content
│
├── Censors
│   ├── add_censor()       → Create learned constraint
│   ├── check_censors()    → Evaluate text against active censors
│   ├── record_false_positive() → Track false triggers
│   ├── escalate()         → Increase severity (warn → block)
│   └── list_censors()     → Active censors for agent
│
├── Working Memory
│   ├── focus()            → Set current task
│   ├── load_item()        → Add item to working memory
│   ├── evict()            → Remove lowest-relevance item
│   ├── get_working_memory() → Current session state
│   └── clear()            → End session, clear working memory
│
├── Unified Recall
│   └── recall()           → Search across ALL memory types, ranked by relevance
│
└── Events
    └── emit_event()       → Log cognitive events to nous_system.events
         │
         └── Reuses: EmbeddingProvider from nous.brain.embeddings
```

## Deliverables

### File Structure

```
nous/heart/
├── __init__.py          # Public exports: Heart, HeartConfig
├── heart.py             # Main Heart class — public API
├── episodes.py          # Episode management
├── facts.py             # Fact storage, confirmation, superseding
├── procedures.py        # K-line management with level-bands
├── censors.py           # Censor registry and evaluation
├── working_memory.py    # Session state management
└── schemas.py           # Pydantic models for Heart inputs/outputs

tests/
├── test_heart.py        # Integration tests for Heart public API
├── test_episodes.py     # Episode lifecycle tests
├── test_facts.py        # Fact CRUD + lifecycle tests
├── test_censors.py      # Censor evaluation + escalation tests
└── test_working_memory.py # Working memory capacity + eviction tests
```

---

### 1. `nous/heart/schemas.py` — Data Transfer Objects

```python
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime


# --- Episodes ---

class EpisodeInput(BaseModel):
    title: str | None = None
    summary: str
    detail: str | None = None
    frame_used: str | None = None
    trigger: str | None = None  # user_message, cron, hook, etc.
    participants: list[str] = []
    tags: list[str] = []

class EpisodeDetail(BaseModel):
    id: UUID
    agent_id: str
    title: str | None
    summary: str
    detail: str | None
    started_at: datetime
    ended_at: datetime | None
    duration_seconds: int | None
    frame_used: str | None
    trigger: str | None
    participants: list[str]
    outcome: str | None
    surprise_level: float | None
    lessons_learned: list[str]
    tags: list[str]
    decision_ids: list[UUID]  # From episode_decisions join
    created_at: datetime

class EpisodeSummary(BaseModel):
    id: UUID
    title: str | None
    summary: str
    outcome: str | None
    started_at: datetime
    tags: list[str]
    score: float | None = None  # Relevance from search


# --- Facts ---

class FactInput(BaseModel):
    content: str
    category: str | None = None  # preference, technical, person, tool, concept, rule
    subject: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str | None = None
    source_episode_id: UUID | None = None
    source_decision_id: UUID | None = None
    tags: list[str] = []

class FactDetail(BaseModel):
    id: UUID
    agent_id: str
    content: str
    category: str | None
    subject: str | None
    confidence: float
    source: str | None
    source_episode_id: UUID | None
    source_decision_id: UUID | None
    learned_at: datetime
    last_confirmed: datetime | None
    confirmation_count: int
    superseded_by: UUID | None
    active: bool
    tags: list[str]
    created_at: datetime

class FactSummary(BaseModel):
    id: UUID
    content: str
    category: str | None
    subject: str | None
    confidence: float
    active: bool
    score: float | None = None


# --- Procedures ---

class ProcedureInput(BaseModel):
    name: str
    domain: str | None = None  # architecture, debugging, deployment, trading, research
    description: str | None = None
    goals: list[str] = []              # Upper fringe
    core_patterns: list[str] = []      # Core
    core_tools: list[str] = []         # Core
    core_concepts: list[str] = []      # Core
    implementation_notes: list[str] = []  # Lower fringe
    tags: list[str] = []

class ProcedureDetail(BaseModel):
    id: UUID
    agent_id: str
    name: str
    domain: str | None
    description: str | None
    goals: list[str]
    core_patterns: list[str]
    core_tools: list[str]
    core_concepts: list[str]
    implementation_notes: list[str]
    activation_count: int
    success_count: int
    failure_count: int
    neutral_count: int
    last_activated: datetime | None
    effectiveness: float | None  # success_count / (success + failure) if > 0
    tags: list[str]
    active: bool
    created_at: datetime

class ProcedureSummary(BaseModel):
    id: UUID
    name: str
    domain: str | None
    activation_count: int
    effectiveness: float | None
    score: float | None = None


# --- Censors ---

class CensorInput(BaseModel):
    trigger_pattern: str
    reason: str
    action: str = "warn"  # warn, block, absolute
    domain: str | None = None
    learned_from_decision: UUID | None = None
    learned_from_episode: UUID | None = None

class CensorDetail(BaseModel):
    id: UUID
    agent_id: str
    trigger_pattern: str
    action: str
    reason: str
    domain: str | None
    learned_from_decision: UUID | None
    learned_from_episode: UUID | None
    created_by: str
    activation_count: int
    last_activated: datetime | None
    false_positive_count: int
    escalation_threshold: int
    active: bool
    created_at: datetime

class CensorMatch(BaseModel):
    """Result from check_censors — a censor that matched."""
    id: UUID
    trigger_pattern: str
    action: str  # warn, block, absolute
    reason: str
    domain: str | None


# --- Working Memory ---

class WorkingMemoryItem(BaseModel):
    type: str  # fact, procedure, decision, censor, episode
    ref_id: UUID
    summary: str
    relevance: float = Field(ge=0.0, le=1.0)
    loaded_at: datetime

class OpenThread(BaseModel):
    description: str
    decision_id: UUID | None = None
    priority: str = "medium"  # low, medium, high
    created_at: datetime

class WorkingMemoryState(BaseModel):
    agent_id: str
    session_id: str
    current_task: str | None
    current_frame: str | None
    items: list[WorkingMemoryItem]
    open_threads: list[OpenThread]
    max_items: int
    item_count: int


# --- Unified Recall ---

class RecallResult(BaseModel):
    """A single result from unified recall across memory types."""
    type: str  # episode, fact, procedure, censor
    id: UUID
    summary: str
    score: float
    metadata: dict = {}  # Type-specific fields
```

---

### 2. `nous/heart/episodes.py` — Episode Management

```python
class EpisodeManager:
    """Manages episodic memory — what happened."""

    def __init__(self, db: Database, embeddings: EmbeddingProvider | None, agent_id: str): ...

    async def start(self, input: EpisodeInput) -> EpisodeDetail:
        """Start a new episode. Sets started_at to now.
        Generate embedding from f"{input.title or ''} {input.summary}".
        Insert into heart.episodes.
        Return full EpisodeDetail.
        """

    async def end(self, episode_id: UUID, outcome: str, lessons_learned: list[str] | None = None,
                  surprise_level: float | None = None) -> EpisodeDetail:
        """Close an episode. Set ended_at, duration_seconds, outcome, lessons_learned, surprise_level.
        Duration = ended_at - started_at in seconds.
        """

    async def link_decision(self, episode_id: UUID, decision_id: UUID) -> None:
        """Insert into heart.episode_decisions."""

    async def link_procedure(self, episode_id: UUID, procedure_id: UUID, effectiveness: str | None = None) -> None:
        """Insert into heart.episode_procedures with optional effectiveness rating."""

    async def get(self, episode_id: UUID) -> EpisodeDetail | None:
        """Fetch episode with linked decision_ids via episode_decisions join."""

    async def list_recent(self, limit: int = 10, outcome: str | None = None) -> list[EpisodeSummary]:
        """List recent episodes ordered by started_at DESC. Optional outcome filter."""

    async def search(self, query: str, limit: int = 10) -> list[EpisodeSummary]:
        """Hybrid search (vector + keyword) over episodes.
        Same CTE pattern as Brain.query() but against heart.episodes.
        Vector weight 0.7, keyword weight 0.3.
        """
```

---

### 3. `nous/heart/facts.py` — Semantic Memory

```python
class FactManager:
    """Manages semantic memory — what we know."""

    def __init__(self, db: Database, embeddings: EmbeddingProvider | None, agent_id: str): ...

    async def learn(self, input: FactInput) -> FactDetail:
        """Store a new fact.
        1. Generate embedding from content
        2. Check for near-duplicates (cosine similarity > 0.95 with active facts)
           - If duplicate found, call confirm() on existing fact instead
           - Return the existing fact (not a new one)
        3. Insert into heart.facts
        4. Return FactDetail
        """

    async def confirm(self, fact_id: UUID) -> FactDetail:
        """Confirm a fact is still true.
        Update last_confirmed = now(), confirmation_count += 1.
        """

    async def supersede(self, old_fact_id: UUID, new_fact: FactInput) -> FactDetail:
        """Replace a fact with a newer version.
        1. Store new fact via learn()
        2. Update old fact: superseded_by = new_fact.id, active = false
        3. Return the new fact
        """

    async def contradict(self, fact_id: UUID, contradicting_fact: FactInput) -> FactDetail:
        """Store a fact that contradicts an existing one.
        1. Store new fact via learn()
        2. Update new fact: contradiction_of = fact_id
        3. Reduce confidence of old fact by 0.2 (min 0.0)
        4. Return the new fact
        """

    async def get(self, fact_id: UUID) -> FactDetail | None:
        """Fetch a single fact."""

    async def search(self, query: str, limit: int = 10, category: str | None = None,
                     active_only: bool = True) -> list[FactSummary]:
        """Hybrid search over facts. Same CTE pattern as Brain.query().
        Default: only active facts. Set active_only=False to include superseded.
        Optional category filter.
        """

    async def deactivate(self, fact_id: UUID) -> None:
        """Soft-delete a fact. Set active = false."""
```

**Key implementation notes:**
- Near-duplicate detection uses cosine similarity on embeddings
- If no embedding provider, skip dedup (store regardless)
- Confidence decay on contradiction is capped at 0.0
- `superseded_by` creates a chain — follow it to find current version

---

### 4. `nous/heart/procedures.py` — Procedural Memory (K-Lines)

```python
class ProcedureManager:
    """Manages procedural memory — how to do things (K-lines with level-bands)."""

    def __init__(self, db: Database, embeddings: EmbeddingProvider | None, agent_id: str): ...

    async def store(self, input: ProcedureInput) -> ProcedureDetail:
        """Store a new procedure.
        Generate embedding from f"{input.name} {input.description or ''} {' '.join(input.core_patterns)}".
        Insert into heart.procedures.
        """

    async def activate(self, procedure_id: UUID) -> ProcedureDetail:
        """Mark a procedure as activated.
        Update activation_count += 1, last_activated = now().
        Return the procedure.
        """

    async def record_outcome(self, procedure_id: UUID, outcome: str) -> ProcedureDetail:
        """Record procedure activation outcome.
        outcome: 'success' → success_count += 1
                 'failure' → failure_count += 1
                 'neutral' → neutral_count += 1
        """

    async def get(self, procedure_id: UUID) -> ProcedureDetail | None:
        """Fetch procedure with computed effectiveness.
        effectiveness = success_count / (success_count + failure_count) if denominator > 0 else None.
        """

    async def search(self, query: str, limit: int = 10, domain: str | None = None) -> list[ProcedureSummary]:
        """Hybrid search over procedures. Optional domain filter."""

    async def retire(self, procedure_id: UUID) -> None:
        """Retire a procedure (effectiveness too low). Set active = false."""
```

---

### 5. `nous/heart/censors.py` — Censor Registry

```python
class CensorManager:
    """Manages censors — things NOT to do."""

    def __init__(self, db: Database, embeddings: EmbeddingProvider | None, agent_id: str): ...

    async def add(self, input: CensorInput) -> CensorDetail:
        """Create a new censor.
        Generate embedding from f"{input.trigger_pattern} {input.reason}".
        Set created_by = 'manual' (auto_failure and auto_escalation set by event handlers later).
        """

    async def check(self, text: str, domain: str | None = None) -> list[CensorMatch]:
        """Check text against all active censors.
        
        Matching strategy:
        1. Generate embedding for text
        2. Find censors with cosine similarity > 0.7 to the text
        3. Optionally filter by domain
        4. For each match: increment activation_count, update last_activated
        5. Check escalation: if activation_count >= escalation_threshold AND action == 'warn',
           auto-escalate to 'block'
        6. Return list of CensorMatch ordered by similarity (highest first)
        
        If no embedding provider: fall back to keyword matching using tsvector search
        against trigger_pattern.
        """

    async def record_false_positive(self, censor_id: UUID) -> CensorDetail:
        """Record a false positive trigger.
        Update false_positive_count += 1, last_false_positive = now().
        If false_positive_count > activation_count * 0.5 (more than half are false):
            consider deactivating (log warning, don't auto-deactivate).
        """

    async def escalate(self, censor_id: UUID) -> CensorDetail:
        """Manually escalate censor severity.
        warn → block → absolute. No downgrade.
        """

    async def list_active(self, domain: str | None = None) -> list[CensorDetail]:
        """List all active censors, optionally filtered by domain."""

    async def deactivate(self, censor_id: UUID) -> None:
        """Deactivate a censor. Set active = false."""
```

**Key implementation notes:**
- Censor matching is semantic (cosine similarity), not regex
- Threshold 0.7 is deliberately lower than Brain's auto_link (0.85) — censors should be sensitive
- Auto-escalation happens during `check()` — the caller doesn't need to trigger it
- Censors don't use the tsvector column (censors table doesn't have one in init.sql), so keyword fallback uses ILIKE on trigger_pattern

---

### 6. `nous/heart/working_memory.py` — Session State

```python
class WorkingMemoryManager:
    """Manages working memory — current session focus."""

    def __init__(self, db: Database, agent_id: str): ...

    async def get_or_create(self, session_id: str) -> WorkingMemoryState:
        """Get existing working memory for session, or create new one.
        Uses UPSERT (INSERT ... ON CONFLICT DO NOTHING) on (agent_id, session_id).
        """

    async def focus(self, session_id: str, task: str, frame: str | None = None) -> WorkingMemoryState:
        """Set the current task and frame.
        Update current_task and current_frame.
        """

    async def load_item(self, session_id: str, item: WorkingMemoryItem) -> WorkingMemoryState:
        """Add an item to working memory.
        If at capacity (items count >= max_items): evict lowest relevance item first.
        Append to items JSONB array.
        Return updated state.
        """

    async def evict(self, session_id: str, ref_id: UUID | None = None) -> WorkingMemoryState:
        """Evict an item from working memory.
        If ref_id provided: remove that specific item.
        If ref_id is None: remove the item with lowest relevance score.
        """

    async def add_thread(self, session_id: str, thread: OpenThread) -> WorkingMemoryState:
        """Add an open thread (pending item)."""

    async def resolve_thread(self, session_id: str, description: str) -> WorkingMemoryState:
        """Remove a thread by matching description (case-insensitive contains)."""

    async def get(self, session_id: str) -> WorkingMemoryState | None:
        """Get current working memory state. Returns None if no session exists."""

    async def clear(self, session_id: str) -> None:
        """Clear working memory for session. DELETE the row."""
```

**Key implementation notes:**
- JSONB items/threads are read-modify-write (load, mutate in Python, write back)
- Capacity enforcement happens in `load_item()` — evict before add if full
- No embeddings needed — working memory is structured, not searched

---

### 7. `nous/heart/heart.py` — Main Heart Class

The public API. Composes all managers.

```python
class Heart:
    """Memory organ for Nous agents."""

    def __init__(self, database: Database, settings: Settings, embedding_provider: EmbeddingProvider | None = None):
        self.db = database
        self.settings = settings
        self.agent_id = settings.agent_id
        self.episodes = EpisodeManager(database, embedding_provider, settings.agent_id)
        self.facts = FactManager(database, embedding_provider, settings.agent_id)
        self.procedures = ProcedureManager(database, embedding_provider, settings.agent_id)
        self.censors = CensorManager(database, embedding_provider, settings.agent_id)
        self.working_memory = WorkingMemoryManager(database, settings.agent_id)
        self._embeddings = embedding_provider
```

#### Public Methods (delegate to managers):

**Episodes:**
- `async def start_episode(self, input: EpisodeInput) -> EpisodeDetail`
- `async def end_episode(self, episode_id: UUID, outcome: str, ...) -> EpisodeDetail`
- `async def get_episode(self, episode_id: UUID) -> EpisodeDetail | None`
- `async def list_episodes(self, limit: int = 10, ...) -> list[EpisodeSummary]`
- `async def link_decision_to_episode(self, episode_id: UUID, decision_id: UUID) -> None`
- `async def link_procedure_to_episode(self, episode_id: UUID, procedure_id: UUID, effectiveness: str | None = None) -> None`

**Facts:**
- `async def learn(self, input: FactInput) -> FactDetail`
- `async def confirm_fact(self, fact_id: UUID) -> FactDetail`
- `async def supersede_fact(self, old_id: UUID, new_fact: FactInput) -> FactDetail`
- `async def contradict_fact(self, fact_id: UUID, new_fact: FactInput) -> FactDetail`
- `async def get_fact(self, fact_id: UUID) -> FactDetail | None`
- `async def search_facts(self, query: str, ...) -> list[FactSummary]`

**Procedures:**
- `async def store_procedure(self, input: ProcedureInput) -> ProcedureDetail`
- `async def activate_procedure(self, procedure_id: UUID) -> ProcedureDetail`
- `async def record_procedure_outcome(self, procedure_id: UUID, outcome: str) -> ProcedureDetail`
- `async def get_procedure(self, procedure_id: UUID) -> ProcedureDetail | None`
- `async def search_procedures(self, query: str, ...) -> list[ProcedureSummary]`

**Censors:**
- `async def add_censor(self, input: CensorInput) -> CensorDetail`
- `async def check_censors(self, text: str, domain: str | None = None) -> list[CensorMatch]`
- `async def record_false_positive(self, censor_id: UUID) -> CensorDetail`
- `async def escalate_censor(self, censor_id: UUID) -> CensorDetail`
- `async def list_censors(self, domain: str | None = None) -> list[CensorDetail]`

**Working Memory:**
- `async def focus(self, session_id: str, task: str, frame: str | None = None) -> WorkingMemoryState`
- `async def load_to_working_memory(self, session_id: str, item: WorkingMemoryItem) -> WorkingMemoryState`
- `async def get_working_memory(self, session_id: str) -> WorkingMemoryState | None`
- `async def clear_working_memory(self, session_id: str) -> None`

**Unified Recall:**
- `async def recall(self, query: str, limit: int = 10, types: list[str] | None = None) -> list[RecallResult]`

#### Unified Recall Implementation

```python
async def recall(self, query: str, limit: int = 10, types: list[str] | None = None) -> list[RecallResult]:
    """Search across ALL memory types, return ranked results.
    
    1. If types is None, search all: episodes, facts, procedures, censors
    2. Run parallel searches (limit * 2 each to have enough for merging)
    3. Normalize scores to [0, 1] range
    4. Merge and sort by score DESC
    5. Return top `limit` results
    
    Each result includes:
    - type: which memory store it came from
    - id: the record UUID
    - summary: human-readable summary
    - score: relevance score
    - metadata: type-specific extra fields
    """
```

---

### 8. Events

Heart emits these events to `nous_system.events`:

| Event Type | When | Data |
|-----------|------|------|
| `episode_started` | start_episode() | `{episode_id}` |
| `episode_completed` | end_episode() | `{episode_id, outcome, duration}` |
| `fact_learned` | learn() | `{fact_id, category, subject}` |
| `fact_confirmed` | confirm_fact() | `{fact_id, confirmation_count}` |
| `fact_superseded` | supersede_fact() | `{old_fact_id, new_fact_id}` |
| `procedure_activated` | activate_procedure() | `{procedure_id}` |
| `procedure_outcome` | record_procedure_outcome() | `{procedure_id, outcome}` |
| `censor_created` | add_censor() | `{censor_id, trigger, action}` |
| `censor_triggered` | check_censors() | `{censor_id, matched_text}` |
| `censor_escalated` | escalate() or auto-escalation | `{censor_id, old_action, new_action}` |

---

### 9. Tests

#### `tests/test_heart.py` — Integration Tests (Heart public API)

1. **`test_full_episode_lifecycle`** — start → link decision → link procedure → end with outcome
2. **`test_learn_and_recall`** — learn 3 facts, recall by query, verify ranking
3. **`test_fact_deduplication`** — learn same fact twice, second call confirms instead of creating new
4. **`test_supersede_fact`** — supersede old fact, verify chain and active flags
5. **`test_contradict_fact`** — contradict, verify confidence reduction
6. **`test_procedure_lifecycle`** — store → activate → record success → check effectiveness
7. **`test_censor_lifecycle`** — add → check triggers → escalate after threshold
8. **`test_working_memory_capacity`** — fill to max, verify eviction of lowest relevance
9. **`test_unified_recall`** — populate all memory types, recall returns mixed results
10. **`test_unified_recall_type_filter`** — recall with types=["fact"] returns only facts
11. **`test_events_emitted`** — verify events logged to nous_system.events

#### `tests/test_episodes.py`

1. **`test_start_episode`** — creates with started_at, no ended_at
2. **`test_end_episode`** — sets ended_at, duration_seconds, outcome
3. **`test_end_episode_calculates_duration`** — duration = ended - started
4. **`test_link_decision`** — episode_decisions row created
5. **`test_link_procedure_with_effectiveness`** — episode_procedures with effectiveness
6. **`test_list_recent`** — ordered by started_at DESC
7. **`test_search_episodes`** — hybrid search returns relevant episodes

#### `tests/test_facts.py`

1. **`test_learn_fact`** — basic creation with all fields
2. **`test_learn_with_provenance`** — source_episode_id and source_decision_id set
3. **`test_confirm_fact`** — confirmation_count increments, last_confirmed updates
4. **`test_supersede_chain`** — A superseded by B, B superseded by C. A and B inactive, C active
5. **`test_contradict_reduces_confidence`** — original fact confidence drops by 0.2
6. **`test_contradict_floor_zero`** — confidence can't go below 0.0
7. **`test_search_active_only`** — superseded facts excluded by default
8. **`test_search_with_category`** — filter by category
9. **`test_deactivate`** — soft delete, search excludes it

#### `tests/test_censors.py`

1. **`test_add_censor`** — creates with correct fields
2. **`test_check_matches`** — censor with similar trigger fires
3. **`test_check_no_match`** — unrelated text doesn't trigger
4. **`test_activation_count_incremented`** — counter goes up on match
5. **`test_auto_escalation`** — after threshold triggers, warn → block
6. **`test_false_positive_tracking`** — count increments
7. **`test_manual_escalation`** — warn → block → absolute
8. **`test_escalation_no_downgrade`** — block cannot go back to warn
9. **`test_inactive_censor_skipped`** — deactivated censors don't match
10. **`test_domain_filter`** — only censors in matching domain trigger

#### `tests/test_working_memory.py`

1. **`test_get_or_create`** — creates new if missing, returns existing if present
2. **`test_focus_sets_task`** — current_task and current_frame updated
3. **`test_load_item`** — item added to items array
4. **`test_capacity_eviction`** — at max_items, lowest relevance evicted before new add
5. **`test_evict_specific`** — remove by ref_id
6. **`test_add_thread`** — thread added to open_threads
7. **`test_resolve_thread`** — thread removed by description match
8. **`test_clear`** — row deleted

---

### 10. Mock Embedding Provider

Reuse the `MockEmbeddingProvider` from `tests/conftest.py` (created in 002). Same fixture, same deterministic embeddings. Add a `mock_heart` fixture:

```python
@pytest_asyncio.fixture
async def heart(db, mock_embeddings):
    """Heart instance with mock embeddings for testing."""
    from nous.heart import Heart
    settings = Settings()
    return Heart(db, settings, embedding_provider=mock_embeddings)
```

---

## Acceptance Criteria

1. `from nous.heart import Heart` works
2. Episodes: start → link decisions/procedures → end with outcome → search
3. Facts: learn with dedup → confirm → supersede chain → contradict with confidence decay
4. Procedures: store with level-bands → activate → record outcome → effectiveness computed
5. Censors: add → semantic check → auto-escalation → false positive tracking
6. Working memory: focus → load items → capacity eviction → clear
7. Unified recall searches across all memory types and returns ranked mixed results
8. All events emitted to `nous_system.events`
9. Embedding provider is optional (keyword-only fallback)
10. All tests pass: `pytest tests/test_heart.py tests/test_episodes.py tests/test_facts.py tests/test_censors.py tests/test_working_memory.py -v`

## Non-Goals (This Phase)

- No auto-extraction pipeline (LLM-based fact extraction from episodes) — requires LLM integration, future phase
- No Brain↔Heart event bus coordination — that's F006 (Event Bus)
- No REST API or MCP endpoints — implementation 004+
- No memory lifecycle automation (auto-trim, auto-archive) — F008
- No context budget system — F005 (Context Engine)
- No summarization tiers — F005

## References

- `docs/features/F002-heart-module.md` — High-level feature spec
- `docs/research/008-database-design.md` — Complete SQL for heart schema
- `docs/research/007-memory-integration.md` — Memory type design rationale
- `docs/research/010-summarization-strategy.md` — Future summarization (not this phase)
- `nous/brain/brain.py` — Brain implementation (pattern to follow)
- `nous/brain/embeddings.py` — Shared EmbeddingProvider
- `nous/storage/models.py` — ORM models (already implemented)

---

*"Memory is reconstruction, not retrieval." — Minsky Ch 8*
