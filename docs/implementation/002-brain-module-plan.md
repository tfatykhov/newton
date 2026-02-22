# 002: Brain Module — Implementation Plan

**Source Spec:** [002-brain-module.md](002-brain-module.md)
**Review Decision IDs:** `b71f3b7f` (arch), `8cf20ee5` (db), `726e1b10` (devil)
**Synthesis Decision ID:** `1d0c1bf0`
**Date:** 2026-02-22

## Review Summary

3-agent review team (nous-arch, nous-db, nous-devil) analyzed the spec from architecture, database, and adversarial angles. High convergence on top 5 issues — all three independently flagged transaction boundaries, score normalization, mock embeddings, emit_event semantics, and bridge_side search gaps.

**Totals after deduplication:** 6 P1 blocking, 15 P2 should-fix, 6 P3 nice-to-have

---

## P1 BLOCKING — Must Fix Before/During Implementation

### P1-1: Transaction Boundaries Unspecified
**Flagged by:** ALL THREE

`record()` has 9 steps with no session/commit guidance. If `auto_link()` (step 8) fails, the decision inserts (steps 4-7) could roll back.

**Resolution:**
- Wrap steps 4-7 (insert decision, tags, reasons, bridge) in a single `async with self.db.session() as session:` block
- Use ORM cascade: single `session.add(decision)` with relationship attributes populated — SQLAlchemy handles cascade inserts
- Isolate `auto_link()` in try/except — failure logs a warning but does NOT roll back the decision
- `emit_event()` runs in the same session (see P2-9)

### P1-2: Test Infrastructure Incompatibility
**Flagged by:** nous-db

Brain creates its own sessions via `self.db.session()`, which bypasses the SAVEPOINT test fixture. Tests write to real DB and don't roll back.

**Resolution:**
- Add optional `session` parameter to all Brain methods that touch the DB:
  ```python
  async def record(self, input: RecordInput, session: AsyncSession | None = None) -> DecisionDetail:
      if session is None:
          async with self.db.session() as session:
              return await self._record(input, session)
      else:
          return await self._record(input, session)
  ```
- Tests inject the fixture session. Production code passes None (auto-creates).

### P1-3: Forward Reference in schemas.py
**Flagged by:** nous-arch

`DecisionDetail` (line 113) references `BridgeInfo` (line 135) and `ThoughtInfo` (line 140) before they're defined. Causes `NameError` at import time.

**Resolution:**
- Add `from __future__ import annotations` at the top of `schemas.py`

### P1-4: MockEmbeddingProvider Produces Degenerate Vectors
**Flagged by:** ALL THREE

SHA-256 hash is 32 bytes, cycled over 1536 dims (`h[i % 32]`). All vectors share the same repeating structure, making cosine similarity meaninglessly high between any pair. `auto_link` and `query` tests become unreliable.

**Resolution:**
```python
class MockEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        import hashlib, random
        h = hashlib.sha256(text.encode()).hexdigest()
        rng = random.Random(h)
        vec = [rng.gauss(0, 1) for _ in range(1536)]
        norm = sum(x*x for x in vec) ** 0.5
        return [x / norm for x in vec]  # L2-normalized

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def close(self) -> None:
        pass
```

### P1-5: Seed Data Test Dependency
**Flagged by:** nous-arch

`test_check_guardrails_blocked` assumes seed.sql guardrails exist, but the SAVEPOINT fixture doesn't load seed data.

**Resolution:**
- Create a `@pytest.fixture` that inserts guardrails directly in the test session:
  ```python
  @pytest_asyncio.fixture
  async def seed_guardrails(session):
      from nous.storage.models import Guardrail
      guardrails = [
          Guardrail(agent_id="nous-default", name="no-high-stakes-low-confidence",
                    condition={"stakes": "high", "confidence_lt": 0.5}, severity="block"),
          # ... other seed guardrails
      ]
      for g in guardrails:
          session.add(g)
      await session.flush()
      return guardrails
  ```

### P1-6: embed() Failure Blocks All Recording
**Flagged by:** nous-devil

If OpenAI API is down, `response.raise_for_status()` propagates an exception through `Brain.record()`. No decisions can be recorded during an outage.

**Resolution:**
- In `Brain.record()` and `Brain.query()`, wrap embedding calls in try/except:
  ```python
  embedding = None
  if self.embeddings:
      try:
          embedding = await self.embeddings.embed(embed_text)
      except Exception:
          logger.warning("Embedding generation failed, recording without embedding")
  ```
- Decision is recorded with `embedding=None`. Query degrades to keyword-only.

---

## P2 SHOULD FIX — Implement During Build

### P2-7: update() Impossible Step
**Flagged by:** nous-arch, nous-devil

Step 3 says "Re-compute quality score if tags/reasons changed" but `update()` only accepts `description`, `context`, `pattern`. Tags/reasons can't change via update.

**Resolution:** Remove step 3 from `update()`. Tags and reasons are immutable after `record()`. Only re-compute quality if the decision is fetched with current tags/reasons and re-scored with updated description/context/pattern fields.

### P2-8: Hybrid Search Score Normalization
**Flagged by:** ALL THREE

`ts_rank_cd` returns unbounded positive floats. Cosine similarity is bounded [0, 1]. Mixing with 0.7/0.3 weights produces scale-mismatched scores.

**Resolution:** Normalize keyword scores:
```sql
ts_rank_cd(search_tsv, plainto_tsquery('english', :query_text)) /
(1.0 + ts_rank_cd(search_tsv, plainto_tsquery('english', :query_text))) AS score
```
This bounds keyword scores to [0, 1).

### P2-9: emit_event Strategy
**Flagged by:** ALL THREE

"Fire-and-forget" contradicts single-transaction. Separate session creates orphan events if main transaction rolls back.

**Resolution:** Include `emit_event` in the main transaction (same session). Accept atomic behavior — if event insert fails, decision insert fails too. Simpler, no orphan risk. True fire-and-forget adds complexity not worth it for v1.

### P2-10: bridge_side Search Unspecified
**Flagged by:** ALL THREE

No SQL, no index, no implementation path for searching bridge columns.

**Resolution:** Implement as `ILIKE` search on bridge structure/function columns. Heuristic search matches the heuristic extraction. Join `brain.decision_bridge` in the query when `bridge_side` is specified:
```sql
JOIN brain.decision_bridge db ON db.decision_id = d.id
WHERE db.{structure|function} ILIKE '%' || :query_text || '%'
```
Add FTS index later if needed.

### P2-11: Brain.close() Missing
**Flagged by:** nous-arch, nous-db

`EmbeddingProvider` holds an `httpx.AsyncClient` that must be closed.

**Resolution:** Implement async context manager on Brain, mirroring Database pattern:
```python
async def close(self) -> None:
    if self.embeddings:
        await self.embeddings.close()

async def __aenter__(self) -> "Brain":
    return self

async def __aexit__(self, *args) -> None:
    await self.close()
```

### P2-12: Config Field Duplication
**Flagged by:** nous-arch, nous-db, nous-devil

`embedding_model` and `embedding_dimensions` already exist in config.py. Spec says to add them again.

**Resolution:** Reference existing fields. Only add new fields:
```python
openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")
auto_link_threshold: float = 0.85
auto_link_max: int = 3
quality_block_threshold: float = 0.5
```

### P2-13: No Literal Types for Enums
**Flagged by:** nous-arch, nous-devil

Category, stakes, outcome, relation are plain `str` — invalid values only caught by DB CHECK constraints with opaque errors.

**Resolution:** Add Literal constraints to Pydantic schemas:
```python
from typing import Literal

CategoryType = Literal["architecture", "process", "tooling", "security", "integration"]
StakesType = Literal["low", "medium", "high", "critical"]
OutcomeType = Literal["pending", "success", "partial", "failure"]
RelationType = Literal["supports", "contradicts", "supersedes", "related_to", "caused_by"]
```

### P2-14: Keyword-Only Scoring Weights
**Flagged by:** nous-devil

In keyword-only fallback mode, combined_score = keyword_score * 0.3. Results look low-relevance.

**Resolution:** When no embedding provider, skip the combined score formula entirely. Use keyword score with weight 1.0:
```python
if self.embeddings:
    # full hybrid with 0.7/0.3 weights
else:
    # keyword-only, score = normalized_keyword_score
```

### P2-15: QualityScorer Class vs Function Mismatch
**Flagged by:** nous-arch

`brain.py` instantiates `self.quality = QualityScorer()` but `quality.py` defines a bare function.

**Resolution:** Define `QualityScorer` as a class wrapping the pure function:
```python
class QualityScorer:
    def compute(self, tags, reasons, pattern, context) -> float:
        return compute_quality_score(tags, reasons, pattern, context)
```

### P2-16: Hybrid Search Filter Clauses
**Flagged by:** nous-db

The CTEs only SELECT id and score — they don't include category/stakes/outcome columns. Filter WHERE clauses can't apply.

**Resolution:** Add filter conditions inside both CTEs as additional WHERE clauses:
```sql
WHERE agent_id = :agent_id AND embedding IS NOT NULL
  AND (:category IS NULL OR category = :category)
  AND (:stakes IS NULL OR stakes = :stakes)
  AND (:outcome IS NULL OR outcome = :outcome)
```

### P2-17: Tag Population in query()
**Flagged by:** nous-db

Spec says "Join with brain.decision_tags to populate tags in results" but the SQL doesn't show it.

**Resolution:** After getting decision IDs from hybrid search, do a separate query for tags:
```python
tag_rows = await session.execute(
    select(DecisionTag).where(DecisionTag.decision_id.in_(decision_ids))
)
tags_by_id = defaultdict(list)
for row in tag_rows.scalars():
    tags_by_id[row.decision_id].append(row.tag)
```

### P2-18: review() Outcome Validation
**Flagged by:** nous-db, nous-devil

No Pydantic validation on outcome parameter. Invalid values hit DB CHECK with opaque error.

**Resolution:** Add input validation:
```python
class ReviewInput(BaseModel):
    outcome: Literal["success", "partial", "failure"]
    result: str | None = None
```

### P2-19: Graph Directionality
**Flagged by:** nous-db

`auto_link()` can create both A->B and B->A edges (redundant for undirected "related_to").

**Resolution:** Normalize edge direction — always use lower UUID as source_id:
```python
if source_id > target_id:
    source_id, target_id = target_id, source_id
```

### P2-20: auto_link() Concurrent Inserts
**Flagged by:** nous-devil

No handling for UNIQUE constraint violation on concurrent edge creation.

**Resolution:** Use `ON CONFLICT DO NOTHING`:
```python
stmt = insert(GraphEdge).values(...).on_conflict_do_nothing(
    index_elements=["source_id", "target_id", "relation"]
)
```

### P2-21: search_tsv Raw SQL Access
**Flagged by:** nous-devil

`search_tsv` is unmapped in ORM (GENERATED ALWAYS). No guidance on accessing it.

**Resolution:** Document in code comment: use `text()` or `literal_column('search_tsv')` for raw SQL queries. The hybrid search already uses raw SQL, so this is consistent.

---

## P3 NICE TO HAVE

1. **Accuracy formula** — Document as "directional agreement", not calibration quality
2. **Session_id in events** — Leave NULL for now, add when session tracking arrives
3. **Guardrail activation_count** — Use SQL-level `activation_count = activation_count + 1` (not read-modify-write)
4. **F001 method name drift** — 002 spec is authoritative (`check` not `check_guardrails`, `get_calibration` not `calibration`)
5. **Negative tests** — Add a few key invalid-input tests (invalid category, confidence > 1.0, nonexistent decision_id)
6. **auto_link threshold** — 0.85 is configurable (already in Settings). Test the config mechanism, not the magic number.

---

## Devil's Advocate Overrides

| Devil's Finding | Severity | Lead Decision | Rationale |
|----------------|----------|--------------|-----------|
| auto_link() race condition | P1 | **Override to P2** | Single-agent deployment initially. ON CONFLICT DO NOTHING is sufficient. |
| search_tsv ORM gap | P1 | **Override to P2** | Spec already uses raw SQL for query(). Just needs documentation. |
| BridgeExtractor value questionable | Challenge | **Override: acceptable** | Explicitly scoped as heuristic-for-now. Upgrade path to LLM extraction is clear. |
| 0.85 threshold untested | Challenge | **Override: acceptable** | It's configurable via Settings. Test the config mechanism, not the number. |

---

## Implementation Phases

### Phase A: Schemas + Pure Functions
**Files:** `schemas.py`, `quality.py`, `bridge.py`
**Dependencies:** None
**Key fixes:** P1-3 (forward refs), P2-13 (Literal types), P2-15 (QualityScorer class)

### Phase B: DB-Dependent Engines
**Files:** `embeddings.py`, `guardrails.py`, `calibration.py`
**Dependencies:** Phase A
**Key fixes:** P1-6 (embed graceful degradation), P2-8 (score normalization), P2-18 (review validation)

### Phase C: Brain Class Integration
**Files:** `brain.py`, `__init__.py`, `config.py` updates
**Dependencies:** Phase A + B
**Key fixes:** P1-1 (transaction boundaries), P1-2 (session injection), P2-7 (update fix), P2-9 (emit_event in-transaction), P2-10 (bridge_side ILIKE), P2-11 (close/context manager), P2-14 (keyword-only weights), P2-16 (search filters), P2-17 (tag population), P2-19 (graph direction), P2-20 (ON CONFLICT)

### Phase D: Tests
**Files:** `test_quality.py`, `test_guardrails.py`, `test_calibration.py`, `test_brain.py`, `conftest.py` updates
**Dependencies:** Phase C
**Key fixes:** P1-4 (mock embeddings), P1-5 (seed data fixtures)

---

## Risk Mitigations

1. **Hybrid search complexity** — Most complex single query. Implement keyword-only first, add vector after.
2. **Embedding API dependency** — Graceful degradation (P1-6) ensures Brain works without OpenAI.
3. **Test isolation** — Session injection (P1-2) is critical; implement in Phase C, test in Phase D.
4. **auto_link in record()** — Isolated in try/except. Decision is always recorded even if linking fails.
5. **Config migration** — Only 4 new fields added to Settings. No breaking changes.

---

*Review team: nous-arch, nous-db, nous-devil. Lead: forge-lead.*
*"Consciousness is menu lists, not deep access." — Minsky Ch 6*
