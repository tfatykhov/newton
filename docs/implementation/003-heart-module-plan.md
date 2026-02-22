# 003: Heart Module — Implementation Plan

**Source Spec:** [003-heart-module.md](003-heart-module.md)
**Review Decision IDs:** `2c7327dc` (arch), `954147ec` (mem), `a94fbe89` (devil)
**Synthesis Decision ID:** `8ed2c6fc`
**Date:** 2026-02-22

## Review Summary

3-agent review team (nous-arch-003, nous-mem-003, nous-devil-003) analyzed the spec from architecture, memory systems, and adversarial perspectives. High convergence on top 5 issues — all three independently flagged session injection, transaction boundaries, and the censor tsvector contradiction. Strong double-convergence on supersede/contradict dedup collision, Heart lifecycle, and mock embedding limitations.

**Totals after deduplication:** 5 P1 blocking, 9 P2 should-fix, 9 P3 nice-to-have

---

## P1 BLOCKING — Must Fix Before/During Implementation

### P1-1: Session Injection Pattern Missing from All Managers + Transaction Boundaries
**Flagged by:** ALL THREE (arch P1-1/P1-2, mem P1-2, devil P1-2/P1-4)

Brain's proven pattern (`brain.py:87-96`) uses `session: AsyncSession | None = None` on every public method with a public/private split. Heart manager constructors take `(db, embeddings, agent_id)` but NO method accepts a session. This breaks:
- SAVEPOINT test isolation (`conftest.py:66-89`)
- Multi-step atomicity (`supersede()` = learn + update old fact, `contradict()` = learn + update confidence)
- Event emission in same transaction as main operation

Without explicit session management, nothing commits — `Database.session()` doesn't auto-commit.

**Resolution:**
- Every manager public method gets `session: AsyncSession | None = None`
- Follow Brain's public/private split:
  ```python
  async def learn(self, input: FactInput, session: AsyncSession | None = None) -> FactDetail:
      if session is None:
          async with self.db.session() as session:
              result = await self._learn(input, session)
              await session.commit()
              return result
      return await self._learn(input, session)
  ```
- Multi-step operations (`supersede`, `contradict`, `check`) pass their session to sub-calls for atomicity
- Tests inject the fixture session; production passes None

### P1-2: supersede()/contradict() + learn() Dedup Collision
**Flagged by:** mem P1-1, devil P1-5

`learn()` checks cosine similarity > 0.95 against active facts. Both `supersede()` and `contradict()` call `learn()` as step 1. Problem: the old fact is still active when `learn()` runs. If the new fact is semantically similar to the old fact (common for superseding/contradicting the same subject), `learn()` will find the old fact as a near-duplicate and call `confirm()` on it instead of creating a new fact.

Calling `contradict(fact_A, similar_content)` would **CONFIRM** fact_A — the exact opposite of intent.

**Resolution:**
- `learn()` accepts an optional `exclude_ids: list[UUID] = []` parameter
- Dedup query adds `WHERE id NOT IN (:exclude_ids)` to skip specific facts
- `supersede(old_id, new_fact)` calls `learn(new_fact, exclude_ids=[old_id])`
- `contradict(fact_id, new_fact)` calls `learn(new_fact, exclude_ids=[fact_id])`

### P1-3: Censor Keyword Fallback Self-Contradiction
**Flagged by:** ALL THREE (arch P2-4, mem P2-2, devil P1-1)

Spec line 462: "fall back to keyword matching using tsvector search against trigger_pattern."
Spec line 489: "Censors don't use the tsvector column (censors table doesn't have one in init.sql), so keyword fallback uses ILIKE on trigger_pattern."

Direct contradiction. Confirmed: `init.sql:310-329` shows no `search_tsv` column on `heart.censors`.

**Resolution:**
- Remove tsvector reference from `check()` docstring
- Keyword fallback uses case-insensitive containment: `lower(trigger_pattern) in lower(input_text)` via `position(lower(censor.trigger_pattern) in lower(text)) > 0` in SQL
- Implementation: `WHERE position(lower(trigger_pattern) in lower(:text)) > 0 AND active = true`

### P1-4: FactDetail Schema Missing contradiction_of Field
**Flagged by:** devil P1-3

`FactDetail` (spec lines 147-163) defines 16 fields but omits `contradiction_of`. However:
- `init.sql:255`: `contradiction_of UUID REFERENCES heart.facts(id)` — column exists
- `models.py:392-394`: `contradiction_of: Mapped[uuid.UUID | None]` — ORM maps it
- Spec `contradict()` line 368: "Update new fact: contradiction_of = fact_id" — method uses it

The schema can't represent data the method produces.

**Resolution:**
- Add `contradiction_of: UUID | None` to `FactDetail` between `superseded_by` and `active`
- Add `contradiction_of: UUID | None = None` to `FactInput` for direct insertion cases

### P1-5: No Read-Only Censor Search for recall()
**Flagged by:** arch P1-3, mem P2-4

`recall()` searches across all memory types. Episodes, facts, and procedures each have a read-only `search()` method. But `CensorManager` only has:
- `check()` — has write side-effects (increments activation_count, updates last_activated, auto-escalates)
- `list_active()` — no semantic search

`recall()` either mutates censor state as a search side-effect, or skips censors entirely.

**Resolution:**
- Add `CensorManager.search(query, limit, domain) -> list[CensorMatch]` — read-only semantic search using the same embedding comparison as `check()` but WITHOUT incrementing counters or auto-escalating
- `recall()` uses `search()` for censors
- `check()` remains the side-effecting method for actual censor evaluation during cognitive processing

---

## P2 SHOULD-FIX — Address During Implementation

### P2-1: Event Emission Location Unspecified
**Flagged by:** arch P2-1, devil P2-5

Spec lists 10 events but never specifies inside/outside transaction. Brain emits events inside the same session (`brain.py:867-879`).

**Resolution:** Managers emit events inside the same session as the main operation. Each manager gets a `_emit_event(session, event_type, data)` helper method modeled on Brain's pattern. The Heart class provides an `Event` model import path.

### P2-2: No Heart.close() or Async Context Manager
**Flagged by:** arch P2-2, devil P2-2

Brain has `close()` + `__aenter__/__aexit__` (`brain.py:72-81`). Heart takes the same `EmbeddingProvider` but has no lifecycle management.

**Resolution:**
- Add `Heart.close()` and `__aenter__/__aexit__` matching Brain's pattern
- Document: the caller (Cognitive Layer or test fixture) owns the `EmbeddingProvider` lifecycle. Neither Brain nor Heart should close a shared provider
- Add `owns_embeddings: bool = True` constructor parameter. Only close the provider if owned

### P2-3: recall() Score Normalization Undefined
**Flagged by:** arch P2-3, mem P2-3

Spec says "Normalize scores to [0, 1] range" without specifying how. Different memory types produce scores from different distributions.

**Resolution:** Use **reciprocal rank fusion (RRF)** instead of raw score normalization. Each sub-search returns ranked results. RRF score = `1 / (k + rank)` where k=60 (standard constant). Merge by RRF score. This avoids the score comparability problem entirely and is well-studied for heterogeneous result merging.

### P2-4: MockEmbeddingProvider Can't Test Similarity Thresholds
**Flagged by:** mem P3-1, devil P2-3

`MockEmbeddingProvider` (SHA-256 seeded PRNG) produces cosine similarity ~0.0 for any different text, ~1.0 for identical text. No way to test the 0.95 dedup threshold or 0.7 censor threshold with "similar but not identical" text.

**Resolution:**
- Add `embed_near(text, noise=0.05)` helper to `MockEmbeddingProvider` that adds controlled noise to produce vectors with predictable cosine similarity
- Tests for dedup use `embed_near` to create vectors in the 0.95-0.99 range
- Tests for censor matching use `embed_near` to create vectors in the 0.7-0.9 range
- Alternative: dedup/censor tests can insert embeddings directly with known cosine distances

### P2-5: Procedures Table Columns Ignored by Spec
**Flagged by:** devil P2-1

`init.sql:285-286`: `related_procedures UUID[]` and `censor_ids UUID[]` exist. ORM `models.py:434-435` maps both. But `ProcedureInput` and `ProcedureDetail` include neither.

**Resolution:** Add both fields to `ProcedureDetail` as optional (they exist in DB and ORM). Do NOT add to `ProcedureInput` — population of these fields is a future feature (cross-linking procedures). Document: "Not populated in this phase — reserved for future procedure cross-referencing."

### P2-6: Raw SQL Templates Needed for Hybrid Search
**Flagged by:** devil P2-4

Spec says "Same CTE pattern as Brain.query()" but `search_tsv` columns are GENERATED ALWAYS and NOT mapped in ORM. Brain uses ~40 lines of raw SQL via `text()` (`brain.py:418-456`). Heart needs 3 separate raw SQL implementations.

**Resolution:** Provide a shared hybrid search helper function that each manager calls with table-specific parameters:
```python
async def _hybrid_search(
    session: AsyncSession,
    table: str,          # "heart.episodes", "heart.facts", etc.
    embedding: list[float],
    query_text: str,
    agent_id: str,
    extra_where: str = "",
    limit: int = 10,
    vector_weight: float = 0.7,
) -> list[tuple[UUID, float]]:
    # Returns (id, combined_score) pairs
```
Place in `nous/heart/search.py` or as a shared utility. Each manager wraps this to load full ORM objects by ID.

### P2-7: Missing Error Handling for Not-Found IDs
**Flagged by:** devil P2-7

No guidance for: `end_episode()` with bad episode_id, `confirm()` with bad fact_id, `activate()` with bad procedure_id, `record_false_positive()` with bad censor_id.

**Resolution:** Follow Brain's pattern — raise `ValueError(f"Episode {episode_id} not found")` when a required entity doesn't exist. Add try/except in Heart wrapper methods if needed.

### P2-8: JSONB Read-Modify-Write Race in Working Memory
**Flagged by:** arch P3-2, mem P2-1

Two concurrent `load_item()` calls could read the same JSONB state, both add an item, and the second write overwrites the first.

**Resolution:** Use `SELECT ... FOR UPDATE` on the working_memory row inside `load_item()`, `evict()`, `add_thread()`, `resolve_thread()`. Acceptable for single-agent sequential use (current scope), but `FOR UPDATE` prevents data loss if concurrent access ever occurs. Document the single-writer assumption.

### P2-9: confirmation_count NULL Handling
**Flagged by:** devil P2-8

ORM `models.py:388`: `confirmation_count: Mapped[int | None]` with `server_default="0"`. Before first DB flush, Python-side value could be None. `confirm()` says `confirmation_count += 1` — incrementing None raises TypeError.

**Resolution:** Use `(fact.confirmation_count or 0) + 1` pattern. Same for all counter fields: `activation_count`, `success_count`, `failure_count`, `neutral_count`, `false_positive_count`.

---

## P3 NICE-TO-HAVE — Implement If Time Allows

### P3-1: ARRAY Column NULL → list Conversion
**Flagged by:** devil P3-1

ORM maps `participants`, `lessons_learned`, `tags`, `goals`, `core_patterns`, etc. as `nullable=True` ARRAY columns. Pydantic schemas expect `list[str]`. DB can return NULL.

**Resolution:** Use `or []` when building Pydantic models from ORM objects: `participants=episode.participants or []`.

### P3-2: Literal Types for Enum-Like String Fields
**Flagged by:** devil P3-2

`WorkingMemoryItem.type`, `CensorInput.action`, outcome values — all plain `str`.

**Resolution:** Add `Literal` types:
```python
from typing import Literal
MemoryType = Literal["fact", "procedure", "decision", "censor", "episode"]
CensorAction = Literal["warn", "block", "absolute"]
EpisodeOutcome = Literal["success", "partial", "failure", "ongoing", "abandoned"]
ProcedureOutcome = Literal["success", "failure", "neutral"]
```

### P3-3: CHECK Constraint Validation in Python
**Flagged by:** devil P3-3

Episode outcome, censor action, procedure outcome — no Python-side validation. Invalid values cause runtime PG CHECK constraint violations.

**Resolution:** With P3-2's Literal types, Pydantic validates automatically. If Literal types not adopted, add explicit validation in manager methods.

### P3-4: Procedure Effectiveness Sample Size
**Flagged by:** mem P3-2

`effectiveness = success_count / (success_count + failure_count)` — 1 success, 0 failures = 100%.

**Resolution:** Use Laplace smoothing: `(success + 1) / (success + failure + 2)`. Defaults to 0.5 for new procedures, converges to true rate with data.

### P3-5: Supersede Chain Traversal Method
**Flagged by:** mem P3-3

Spec says "follow superseded_by to find current version" but provides no method.

**Resolution:** Add `FactManager.get_current(fact_id) -> FactDetail` that follows `superseded_by` chain with max depth=10 and cycle detection. Use recursive CTE:
```sql
WITH RECURSIVE chain AS (
    SELECT id, superseded_by, 1 as depth FROM heart.facts WHERE id = :start_id
    UNION ALL
    SELECT f.id, f.superseded_by, c.depth + 1
    FROM heart.facts f JOIN chain c ON f.id = c.superseded_by
    WHERE c.depth < 10
)
SELECT id FROM chain WHERE superseded_by IS NULL;
```

### P3-6: Working Memory Eviction Tie-Breaking
**Flagged by:** mem P3-4

No rule when multiple items share minimum relevance.

**Resolution:** "When tied, evict the item with the earliest `loaded_at` (FIFO)."

### P3-7: Episode Embedding Not Updated on end_episode()
**Flagged by:** mem P3-5

Embedding generated from `title + summary` at `start_episode()`. When `end_episode()` adds outcome and lessons_learned, embedding is not updated.

**Resolution:** Regenerate embedding on `end_episode()` incorporating outcome and lessons:
```python
embed_text = f"{episode.title or ''} {episode.summary} {outcome} {' '.join(lessons_learned or [])}"
```

### P3-8: Add tests/test_procedures.py
**Flagged by:** devil P4-1, arch P3-3

No dedicated test file for procedures. Missing: search, domain filter, effectiveness computation, retire.

**Resolution:** Create `tests/test_procedures.py` with: test_store, test_search, test_activate, test_record_outcome_effectiveness, test_retire, test_domain_filter.

### P3-9: F002 Coverage Gaps Documentation
**Flagged by:** devil P3-5, P3-6

F002 mentions features not in 003's Non-Goals (three detail levels, shared censor system). F002 interface deviations (`record_episode` vs `start/end_episode`) undocumented.

**Resolution:** Add to Non-Goals section: "Three detail levels per memory (micro/summary/full) — deferred to F005 Context Engine. Brain↔Heart shared censor system — deferred to F006 Event Bus." Note interface improvements over F002 in spec header.

---

## Lead Overrides

1. **Devil P2-9 (episodes no updated_at):** Downgraded from P2 to not included. Episodes are append-mostly (start → end), and `ended_at` serves as the modification timestamp. Adding `updated_at` + trigger is inconsistent with the append model and unnecessary.

2. **Mem P4-1 (Minsky censor/suppressor distinction):** Acknowledged as theoretically correct but explicitly deferred. Current censor model is sufficient for this phase. Suppressors belong in the Cognitive Layer (F005/F006).

3. **Mem P4-2 (circular supersede chains):** Low probability in practice. P3-5's chain traversal includes cycle detection (visited set) which addresses this defensively.

---

## Implementation Phases

### Phase A: Schemas + Shared Utilities
**Files:** `nous/heart/schemas.py`, `nous/heart/search.py`

1. Fix `FactDetail` — add `contradiction_of: UUID | None` (P1-4)
2. Add `ProcedureDetail` fields: `related_procedures`, `censor_ids` (P2-5)
3. Add Literal types for enum fields (P3-2)
4. Add `exclude_ids` to `FactInput` or as separate param (P1-2 prep)
5. Create `nous/heart/search.py` with shared `_hybrid_search()` helper (P2-6)

### Phase B: Manager Classes with Session Injection
**Files:** `nous/heart/episodes.py`, `nous/heart/facts.py`, `nous/heart/procedures.py`, `nous/heart/censors.py`, `nous/heart/working_memory.py`

Every manager method follows Brain's session injection pattern (P1-1).

1. **EpisodeManager** — `start`, `end` (regenerate embedding P3-7), `link_decision`, `link_procedure`, `get`, `list_recent`, `search`
2. **FactManager** — `learn` (with `exclude_ids` P1-2), `confirm` (NULL-safe counter P2-9), `supersede` (passes exclude_ids), `contradict` (passes exclude_ids + adds contradiction_of P1-4), `get`, `search`, `get_current` (P3-5), `deactivate`
3. **ProcedureManager** — `store`, `activate` (NULL-safe counter), `record_outcome` (NULL-safe + Laplace P3-4), `get`, `search`, `retire`
4. **CensorManager** — `add`, `check` (ILIKE fallback P1-3, side-effects only), `search` (read-only P1-5), `record_false_positive`, `escalate`, `list_active`, `deactivate`
5. **WorkingMemoryManager** — `get_or_create`, `focus`, `load_item` (FOR UPDATE P2-8), `evict` (tie-breaking P3-6), `add_thread`, `resolve_thread`, `get`, `clear`

Each manager includes `_emit_event(session, event_type, data)` helper (P2-1).

### Phase C: Heart Class + Unified Recall
**Files:** `nous/heart/heart.py`, `nous/heart/__init__.py`

1. Heart constructor with `owns_embeddings` flag (P2-2)
2. `close()` + `__aenter__/__aexit__` (P2-2)
3. All public delegation methods passing session through
4. `recall()` using RRF normalization (P2-3) with asyncio.gather for parallel sub-searches
5. ValueError for not-found entities (P2-7)
6. `__init__.py` exports: `Heart`, `HeartConfig` (if needed), all schema types

### Phase D: Tests
**Files:** `tests/test_heart.py`, `tests/test_episodes.py`, `tests/test_facts.py`, `tests/test_censors.py`, `tests/test_working_memory.py`, `tests/test_procedures.py`, `tests/conftest.py`

1. Update `conftest.py` — add `embed_near()` to MockEmbeddingProvider (P2-4), add `heart` fixture
2. `test_episodes.py` — 7 tests per spec
3. `test_facts.py` — 9 tests per spec + test for `exclude_ids` dedup bypass
4. `test_procedures.py` — 6+ tests (P3-8)
5. `test_censors.py` — 10 tests per spec + test for read-only `search()`
6. `test_working_memory.py` — 8 tests per spec
7. `test_heart.py` — 11 integration tests per spec

**Dependency order:** A → B → C → D (each phase builds on previous)

---

## Acceptance Criteria Additions

Beyond the original 10 acceptance criteria, the following must also pass:

11. All manager methods accept `session: AsyncSession | None = None` and work with both injected and self-created sessions
12. `supersede()` and `contradict()` never trigger dedup against the fact being replaced
13. Censor keyword fallback uses ILIKE (not tsvector)
14. `FactDetail` includes `contradiction_of` field
15. `recall()` uses reciprocal rank fusion for cross-type ranking
16. `Heart` has `close()` and async context manager support
17. `CensorManager.search()` exists as read-only alternative to `check()`

---

## References

- `docs/implementation/002-brain-module-plan.md` — Precedent plan format and Brain patterns
- `nous/brain/brain.py` — Session injection pattern to replicate
- `sql/init.sql:198-346` — Heart schema (7 tables, source of truth)
- `nous/storage/models.py:300-544` — Heart ORM models
- `tests/conftest.py` — MockEmbeddingProvider and test fixtures

---

*"Memory is reconstruction, not retrieval." — Minsky Ch 8*
