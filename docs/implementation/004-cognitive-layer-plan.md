# 004: Cognitive Layer — Implementation Plan

**Source:** `docs/implementation/004-cognitive-layer.md`
**Review Team:** nous-arch-004, nous-integ-004, nous-devil-004
**Date:** 2026-02-23

## Review Summary

Three-agent review of 004-cognitive-layer spec. High convergence on top issues — all three flagged `brain._db` typo and `Brain.update()` missing `confidence`. Integration reviewer found 12 P1 + 5 P2 API mismatches. Devil's advocate found censor loop risk and async race condition.

**Verdict:** Spec is architecturally sound — 7-phase loop, engine separation, and D1 (no LLM) are well-designed. But 12 API contract mismatches require fixes before implementation. Two require pre-work in Brain module (Brain.update confidence, seed data).

---

## P1 — Must Fix Before Build

### P1-1: `brain._db` → `brain.db` [ALL THREE FLAGGED]

**Spec line 523:** `self._frames = FrameEngine(brain._db, settings)`
**Actual:** Brain exposes `self.db` (public). Will crash with `AttributeError`.
**Fix:** Use `brain.db`.

### P1-2: `Brain.record()` takes `RecordInput`, not kwargs [ARCH + INTEG]

**Spec line 362-367:** Shows keyword args to `Brain.record()`.
**Actual:** `Brain.record(input: RecordInput, session=None)` — single pydantic model.
**Fix:** `DeliberationEngine.start()` must construct:
```python
RecordInput(
    description=f"Plan: {description}",
    confidence=0.5,
    category=frame.default_category or "process",
    stakes=frame.default_stakes or "low",
    tags=[frame.frame_id],
    reasons=[ReasonInput(type="analysis", text=f"Frame '{frame.frame_name}' triggered deliberation for: {description[:100]}")]
)
```
Note: At least 1 reason required — seed guardrail `require-reasons` blocks decisions with `reason_count < 1`.

### P1-3: `Brain.update()` lacks `confidence` parameter [ALL THREE FLAGGED]

**Spec line 382-385:** `finalize()` tries to update confidence from 0.5 → 0.8/0.5/0.3.
**Actual:** `Brain.update(decision_id, description=None, context=None, pattern=None)` — no confidence, no tags, no stakes.
**Fix — PRE-WORK REQUIRED:** Add `confidence: float | None = None` and `tags: list[str] | None = None` to `Brain.update()`. This is a targeted extension of 002-brain-module. Without this, deliberation confidence never updates from initial 0.5.

### P1-4: Censor field is `action`, not `severity` [INTEG]

**Pervasive throughout spec.** Monitor, context formatting, learn() all reference `severity`.
**Actual:** CensorInput/CensorDetail/CensorMatch all use `action` (values: `warn`, `block`, `absolute`).
**Fix:** Replace all `severity` references with `action` during implementation. `_format_censors()` format string: `**{ACTION}:**` not `**{SEVERITY}:**`.

### P1-5: All Heart manager methods take pydantic input models, not kwargs [INTEG]

Affects multiple locations:
| Spec Pattern | Actual Signature |
|---|---|
| `Heart.censors.add(trigger_pattern, reason, severity="warn")` | `CensorManager.add(input: CensorInput)` |
| `Heart.episodes.start(...)` | `EpisodeManager.start(input: EpisodeInput)` |
| `Heart.facts.learn(content, source="reflection")` | `FactManager.learn(input: FactInput)` |

**Fix:** Construct pydantic models:
```python
# Censor creation (in MonitorEngine.learn)
CensorInput(trigger_pattern=text, reason="Auto-created from tool error", action="warn")

# Episode start (in CognitiveLayer.pre_turn)
EpisodeInput(summary=user_input[:200], frame_used=frame.frame_id, trigger="user_message")

# Fact learning (in CognitiveLayer.end_session)
FactInput(content=learned_text, source="reflection", category="rule")
```

### P1-6: `Heart.recall()` param is `types` (list), not `type` (str) [INTEG]

**Spec line 330:** `Heart.recall(input_text, type="fact")`
**Actual:** `Heart.recall(query, limit=10, types=["fact"])`
**Fix:** Use `types=["fact"]`, `types=["procedure"]`, `types=["episode"]`.

**Architectural note (ARCH P2-2):** Consider using type-specific search methods instead (`Heart.search_facts()`, `Heart.search_procedures()`, `Heart.search_episodes()`). These return typed results (FactSummary, ProcedureSummary, EpisodeSummary) that are easier to format. Reserve `Heart.recall()` for unified cross-type ranking. **Decision for implementer: either approach works — type-specific is cleaner for formatting.**

### P1-7: `WorkingMemoryManager.get()` does not take `agent_id` [INTEG]

**Spec line 333:** `Heart.working_memory.get(agent_id, session_id)`
**Actual:** `WorkingMemoryManager.get(session_id, session=None)` — agent_id is bound at construction.
**Fix:** Call `Heart.working_memory.get(session_id)` or `Heart.get_working_memory(session_id)`.

Also: Must call `get_or_create_working_memory(session_id)` before `focus()` on first turn — `focus()` raises `ValueError` if no WM row exists.

### P1-8: `EpisodeManager.end()` has no `summary` parameter [INTEG]

**Spec line 467-468:** `summary: f"{frame.frame_name}: {turn_result.response_text[:200]}"`
**Actual:** `EpisodeManager.end(episode_id, outcome, lessons_learned=None, surprise_level=None)` — summary is set at `start()` time only.
**Fix:** Drop summary from `end()` call. Set a good summary at `start()` time instead.

### P1-9: `default_stakes` not in seed data [ARCH + INTEG]

**Spec line 365:** `frame.default_stakes or "low"` — but all frames have `default_stakes=NULL`.
**Fix — PRE-WORK REQUIRED:** Update `seed.sql` to include `default_stakes`:
```sql
-- Add to seed.sql INSERT:
-- task: 'medium', question: 'low', decision: 'high', creative: 'low', conversation: 'low', debug: 'medium'
```

### P1-10: `DecisionSummary` has no `reasons` field [INTEG + ARCH]

**Spec line 261:** `_format_decisions()` says "Reasons: {comma-separated reason texts}"
**Actual:** `Brain.query()` returns `DecisionSummary` which has no `reasons` field (only on `DecisionDetail`).
**Fix:** Drop reasons from `_format_decisions()` format. Use: `- [{outcome or "pending"}] {description} (confidence: {confidence})`. Reasons add little value in a truncated context section and would require N extra `Brain.get()` calls.

---

## P2 — Should Fix During Build

### P2-1: Multi-word activation patterns broken by word tokenization [ARCH]

Seed data has multi-word patterns: `'what if'`, `'how are you'`, `'tell me'`, `'trade-off'`. Word tokenization won't match these.
**Fix:** Check multi-word patterns as substrings, single-word as set membership:
```python
for pattern in frame.activation_patterns:
    if ' ' in pattern:
        if pattern in input_lower:
            match_count += 1
    else:
        if pattern in input_words:
            match_count += 1
```

### P2-2: Episode lifecycle — `learn()` ends episode, stale in tracking dict [ARCH]

`post_turn` → `learn()` ends episode every turn. Next `pre_turn` sees session in `_active_episodes` (stale), skips starting new episode. Subsequent turns reference ended episode.
**Fix:** Do NOT end episodes in `learn()`. Only end in `end_session()`. One episode per session. Remove episode ending from `MonitorEngine.learn()`.

### P2-3: Surprise heuristic false positives [DEVIL]

Checking "failed"/"error"/"couldn't" in response text fires on legitimate content like "The error was caused by...". Debug frames almost always contain "error".
**Fix:** Only match at sentence/phrase level, or check `turn_result.error is not None` (structural) before falling back to text matching. Suggested approach:
- 0.9 if `turn_result.error is not None` (structural — always correct)
- 0.3 if any `tool_result.error` exists (structural — always correct)
- Skip text-matching for v0.1 — too noisy. Revisit with semantic analysis later.

### P2-4: Censor creation feedback loop — no circuit breaker [DEVIL]

Tool error → censor → agent avoids tool → uses alternative → alternative fails → more censors.
**Fix:** Deduplicate before creating: check if censor with same `trigger_pattern` already exists for this agent. Also cap at max 3 auto-created censors per session.

### P2-5: Per-section error isolation in `ContextEngine.build()` [DEVIL]

If one section's data fetch fails, don't throw away everything.
**Fix:** Wrap each section in try/except:
```python
try:
    decisions = await self._brain.query(input_text, limit=5, session=session)
except Exception:
    logger.warning("Brain.query failed during context build")
    decisions = []
```

### P2-6: UUID string conversion not specified [DEVIL]

`TurnContext.decision_id` is `str | None`. `Brain.record()` returns `DecisionDetail` with `id: UUID`. `Brain.think()` takes `UUID`. Conversions needed throughout.
**Fix:** Use `str(uuid)` when storing in TurnContext, `UUID(decision_id)` when calling Brain methods. Document in each method.

### P2-7: Delegate event emission to `Brain.emit_event()` [ARCH]

CognitiveLayer shouldn't import Event ORM directly — layer violation.
**Fix:** Use `Brain.emit_event(event_type, data, session)`. Include `session_id` in the `data` dict as a workaround (Brain.emit_event doesn't accept session_id directly).

### P2-8: `FactSummary` lacks `confirmation_count`/`status`, `ProcedureSummary` lacks `core_patterns` [INTEG]

Format methods reference fields not available on summary types.
**Fix for facts:** Use `- {content} [confidence: {confidence}, {'active' if active else 'inactive'}]`
**Fix for procedures:** Use `- **{name}** ({domain}): {description[:100]}` — or call `Heart.get_procedure()` for full detail if budget allows.

### P2-9: Reflection parsing contract undefined [DEVIL]

"learned: X" parsing is fragile — no case/format specification.
**Fix:** Define regex: `r'^\s*[-*]?\s*learned:\s*(.+)$'` (case-insensitive, supports markdown bullets). Add tests for various formats.

### P2-10: `_active_episodes` race condition at await boundaries [DEVIL]

Two coroutines calling `pre_turn` for same session_id can create duplicate episodes.
**Fix for v0.1:** Document as known limitation (single agent, single runtime). Add comment noting the race window. Full fix with `asyncio.Lock` per session can come later.

---

## P3 — Nice to Have (implement if time allows)

| # | Issue | Fix |
|---|-------|-----|
| P3-1 | No logging in any module | Add `logger = logging.getLogger(__name__)` to each file |
| P3-2 | No metrics/timing | Add `duration_ms` to `turn_completed` event data |
| P3-3 | DB connection drop unhandled | Covered by P2-5 per-section isolation |
| P3-4 | `ContextBudget.for_frame()` no validation | Accept raw string, fallback is fine |
| P3-5 | Token estimation rough (4 chars/token) | Acceptable for v0.1 |
| P3-6 | D11 contradiction surfacing not implemented | Remove reference or defer to future spec |
| P3-7 | `questions_to_ask` column may be missing from Frame model | Verify — ORM has it, seed data doesn't populate |
| P3-8 | Test gap: concurrent pre_turn same session | Add if P2-10 is fixed |
| P3-9 | Test gap: post_turn without prior pre_turn | Add graceful degradation test |
| P3-10 | Test gap: end_session without pre_turn | Use `.pop(session_id, None)` |

---

## Pre-Work Required (before 004 implementation)

These changes touch 002-brain-module code and seed data:

### 1. Extend `Brain.update()` signature

In `nous/brain/brain.py`, add `confidence` and `tags` parameters:
```python
async def update(
    self,
    decision_id: UUID,
    description: str | None = None,
    context: str | None = None,
    pattern: str | None = None,
    confidence: float | None = None,  # NEW
    tags: list[str] | None = None,     # NEW
    session: AsyncSession | None = None,
) -> DecisionDetail:
```

Internal `_update` must handle:
- `confidence`: set `decision.confidence = confidence` if not None
- `tags`: replace existing tags (delete old DecisionTag rows, insert new)

### 2. Update `seed.sql` with `default_stakes`

```sql
INSERT INTO nous_system.frames (id, agent_id, name, description, activation_patterns, default_category, default_stakes) VALUES
    ('task', 'nous-default', 'Task Execution', 'Focused on completing a specific task', ARRAY['build', 'fix', 'create', 'implement', 'deploy'], 'tooling', 'medium'),
    ('question', 'nous-default', 'Question Answering', 'Answering questions, looking things up', ARRAY['what', 'how', 'why', 'explain', 'tell me'], 'process', 'low'),
    ('decision', 'nous-default', 'Decision Making', 'Evaluating options, choosing a path', ARRAY['should', 'choose', 'decide', 'compare', 'trade-off'], 'architecture', 'high'),
    ('creative', 'nous-default', 'Creative', 'Brainstorming, ideation, exploration', ARRAY['imagine', 'brainstorm', 'what if', 'design', 'explore'], 'architecture', 'low'),
    ('conversation', 'nous-default', 'Conversation', 'Casual or social interaction', ARRAY['hello', 'hi', 'thanks', 'how are you'], 'process', 'low'),
    ('debug', 'nous-default', 'Debug', 'Investigating problems, tracing errors', ARRAY['error', 'bug', 'broken', 'failing', 'crash', 'wrong'], 'tooling', 'medium');
```

---

## Implementation Phases

### Phase A: Pre-work (Brain.update extension + seed data)
- Extend `Brain.update()` with `confidence` and `tags` params
- Update `seed.sql` with `default_stakes` values
- Add tests for new Brain.update params
- **Assignee:** db-engineer or python-engineer

### Phase B: Schemas + FrameEngine
- `nous/cognitive/schemas.py` — pydantic models (as spec, no changes needed)
- `nous/cognitive/frames.py` — frame selection with P2-1 multi-word fix
- `tests/test_frames.py`
- **Assignee:** python-engineer

### Phase C: ContextEngine
- `nous/cognitive/context.py` — context assembly
- Apply P1-6 (types list), P1-7 (no agent_id on WM), P1-10 (drop reasons from format), P2-5 (per-section isolation), P2-8 (format field fixes)
- Use type-specific search methods (search_facts, search_procedures, search_episodes) for cleaner typing
- `tests/test_context.py`
- **Assignee:** python-engineer

### Phase D: DeliberationEngine
- `nous/cognitive/deliberation.py` — deliberation lifecycle
- Apply P1-2 (RecordInput construction), P1-3 (confidence update via updated Brain.update), P2-6 (UUID conversions)
- `tests/test_deliberation.py`
- **Assignee:** python-engineer

### Phase E: MonitorEngine
- `nous/cognitive/monitor.py` — assessment + learning
- Apply P1-4 (action not severity), P1-5 (CensorInput construction), P1-8 (no summary on end), P2-2 (don't end episode in learn), P2-3 (drop text-matching heuristic), P2-4 (deduplicate censors)
- `tests/test_monitor.py`
- **Assignee:** python-engineer

### Phase F: CognitiveLayer + Integration Tests
- `nous/cognitive/layer.py` — main orchestrator
- `nous/cognitive/__init__.py` — public exports
- Apply P1-1 (brain.db), P2-7 (delegate to Brain.emit_event), P2-9 (reflection regex), P2-10 (document race condition)
- `tests/test_cognitive_layer.py` — full integration tests
- **Assignee:** python-engineer

### Phase G: Code Review
- Independent review of all Phase A-F code
- Verify all P1/P2 fixes are applied
- Security check (no SQL injection, input validation)
- Test coverage verification
- **Assignee:** code-reviewer

---

## Implementation Team

| Agent | Role | Phases | Focus |
|-------|------|--------|-------|
| **python-engineer** | Primary implementer | A-F | All production code + tests |
| **test-engineer** | Test specialist | B-F (parallel) | Write tests from spec, verify coverage |
| **code-reviewer** | Independent reviewer | G | Quality, security, completeness |

Pipeline: python-engineer implements (A→B→C→D→E→F) → code-reviewer reviews (G) → full test suite run

---

## Acceptance Criteria

1. All 7 files created in `nous/cognitive/`
2. All 5 test files pass
3. `Brain.update()` accepts `confidence` and `tags` (pre-work)
4. `seed.sql` updated with `default_stakes`
5. No P1 issues remain
6. Full test suite passes: `pytest tests/ -v`
