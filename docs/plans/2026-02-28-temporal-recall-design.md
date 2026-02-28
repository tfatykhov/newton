# Design: 008.6 Temporal Recall

**Date:** 2026-02-28
**Spec:** docs/implementation/008.6-temporal-recall.md
**Status:** Approved

## Problem

When a user asks "what did we talk about recently?", Nous uses semantic search which matches by meaning similarity, not time. Cross-domain topics (like a ski trip) have zero semantic overlap with a recap question — so they're invisible. Additionally, `list_recent()` filters `active=True`, but after 008.3 all closed episodes have `active=False`, returning empty results.

## Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Tool design | Separate `recall_recent` tool | Clearer intent than extending `recall_deep`. Agent explicitly chooses time-based vs semantic recall. |
| Recap detection | Both patterns + intent signals | Pattern matching for explicit recap queries, temporal_recency signals for broader time-awareness. Belt and suspenders. |
| Temporal context | Always-on | ~50 tokens/turn cost. Eliminates blind spots entirely rather than just reducing them. Configurable via toggle. |
| Auto-inject vs boost | Budget boost (Approach B) | Recap detection boosts existing context assembly budget instead of separate injection path. Single mechanism, no duplicate code paths. |
| Frame access | All frames (like recall_deep) | Consistent with existing pattern. No downside — agent ignores it when irrelevant. |
| Approach | B — Unified Context | Single mechanism (context assembly) handles both always-on and recap. 3-tier escalation: titles → summaries → explicit tool. |

## Architecture

### 3-Tier Escalation

| Tier | Trigger | What's Included | Tokens |
|------|---------|----------------|--------|
| 1 (always) | Every turn | Episode titles + timestamps | ~50 |
| 2 (boost) | Recap patterns or temporal_recency > 0.5 | Titles + summaries, doubled budget | ~200-300 |
| 3 (explicit) | Agent calls `recall_recent` | Full formatted output | Agent-controlled |

### Recall Flow (After 008.6)

```
User query → Intent classification (temporal_recency signals)
          → Recap pattern check
          → Context assembly:
              - Always-on: recent episode titles (Tier 1)
              - If recap/temporal: boosted episode budget + summaries (Tier 2)
          → Agent responds (may call recall_recent for Tier 3)
```

## Change 1: Fix `list_recent()` Bug

**File:** `nous/heart/episodes.py`

Change `WHERE active = True` to `WHERE ended_at IS NOT NULL`. Returns completed episodes, not ongoing ones. Add optional `hours` parameter for time-windowed queries.

```python
async def _list_recent(self, limit, outcome, session, hours=None):
    stmt = (
        select(Episode)
        .where(Episode.agent_id == self.agent_id)
        .where(Episode.ended_at.isnot(None))  # Completed episodes only
        .order_by(Episode.started_at.desc())
        .limit(limit)
    )
    if hours:
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        stmt = stmt.where(Episode.started_at >= cutoff)
```

## Change 2: `recall_recent` Tool

**File:** `nous/api/tools.py`

New tool registered alongside `recall_deep`. Available in all frames. Takes `hours` (default 48) and `limit` (default 10). Returns formatted list of recent episodes with titles, timestamps, and summaries.

## Change 3: Always-On Temporal Context Tier

**File:** `nous/cognitive/context.py`

Insert before semantic episode search (Priority 7.5). Always includes recent episode titles + timestamps. ~50 tokens for 5 episodes. Gated by `NOUS_TEMPORAL_CONTEXT_ENABLED` config toggle.

Deduplication: episodes shown in temporal tier are excluded from semantic episode search by ID.

## Change 4: Recap Detection + Budget Boost

**Files:** `nous/cognitive/layer.py` + `nous/cognitive/intent.py`

Recap patterns (frozenset) detect explicit recap queries. Existing `temporal_recency` signals from intent classifier wired into retrieval plan. When recap detected OR temporal_recency > 0.5:
- Double episode token budget
- Include summaries instead of just titles in temporal tier
- Sort by time (not relevance score)

## Files Changed

| File | Change | Lines (est.) |
|------|--------|-------------|
| `nous/heart/episodes.py` | Fix `list_recent()` active filter, add `hours` param | ~5 |
| `nous/api/tools.py` | Register `recall_recent` tool | ~30 |
| `nous/api/runner.py` | Add `recall_recent` to `FRAME_TOOLS` (all frames) | ~2 |
| `nous/cognitive/context.py` | Temporal tier + dedup + boost support | ~25 |
| `nous/cognitive/layer.py` | Recap detection + temporal_boost flag | ~20 |
| `nous/cognitive/intent.py` | Wire temporal_recency into retrieval plan | ~5 |
| `nous/config.py` | `NOUS_TEMPORAL_CONTEXT_ENABLED` toggle | ~2 |
| `tests/test_episodes.py` | Fix list_recent tests, add hours param tests | ~15 |
| `tests/test_temporal_recall.py` | Tool, context tier, recap detection, budget boost | ~60 |

**Total:** ~155 lines. No schema changes, no migrations.

## Phased Delivery

| Phase | What Ships | Lines (est.) |
|-------|-----------|-------------|
| Phase 1 | Fix `list_recent()` bug + tests | ~15 |
| Phase 2 | Always-on temporal context tier + config toggle + tests | ~40 |
| Phase 3 | `recall_recent` tool + registration + frame access + tests | ~50 |
| Phase 4 | Recap detection + intent signal wiring + budget boost + tests | ~50 |

Phase 1 must go first. Phases 2-4 are independent of each other but build on Phase 1.

## Not In Scope

- Full conversation replay (use existing `recall_deep` for deep dives)
- Calendar-style date queries ("what happened on Tuesday") — needs date parsing
- Cross-user temporal recall (F016 multi-agent scope)
- Schema changes or migrations
