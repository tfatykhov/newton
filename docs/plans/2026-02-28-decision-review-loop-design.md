# Design: 008.5 Decision Review Loop

**Date:** 2026-02-28
**Spec:** docs/implementation/008.5-decision-review-loop.md
**Status:** Approved

## Problem

Nous records decisions but never reviews them. 30 decisions exist, 0 reviewed, 0 calibration snapshots. The Brain module is write-only. Without review, confidence scores are unvalidated, bad patterns repeat, and F007 metrics have no data.

## Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Session linking | Add `session_id` column on decisions | Denormalized but direct. One WHERE clause vs joining through episode_decisions. |
| Signal scope | All 4 signals (Error, Episode, FileExists, GitHub) | Full coverage from day one. GitHubSignal skips gracefully if no token. |
| Tier 2 API | REST endpoint `POST /decisions/{id}/review` | Clean API boundary. Works for any external reviewer. |
| Daily sweep | Session-end piggyback + standalone `sweep()` | No scheduler exists yet. `sweep()` designed for future scheduler migration (one-line change). |
| Tier 3 escalation | External agent responsibility | Nous exposes `GET /decisions/unreviewed`. Another agent queries and sends Telegram notifications. |
| Reviewer tracking | Extend `brain.review()` with `reviewer` param | Single source of truth. Auto-review passes "auto", REST passes caller identity. |
| Signal architecture | Protocol-based (Approach A) | Matches existing patterns (handlers, guardrails). Easy to extend. Spec's design. |

## Architecture

### Three-Tier Review System

- **Tier 1 (Automated):** `DecisionReviewer` handler fires on `session_ended`. Runs 4 signal checks against session decisions + sweeps older unreviewed decisions.
- **Tier 2 (Cross-Agent):** External agents (Emerson) call `POST /decisions/{id}/review` to review decisions Tier 1 couldn't resolve.
- **Tier 3 (Human Escalation):** External agent queries `GET /decisions/unreviewed?stakes=high&max_age_days=14`, sends summary to Tim via Telegram. Not built in Nous.

## Schema Changes

Migration: `sql/migrations/009_decision_review.sql`

| Column | Type | Table | Purpose |
|--------|------|-------|---------|
| `session_id` | `VARCHAR(100)` | `brain.decisions` | Links decision to session. Nullable (backfill-safe). |
| `reviewer` | `VARCHAR(50)` | `brain.decisions` | Who reviewed: auto, emerson, tim, etc. Nullable. |

No new tables.

## Signal System

Protocol-based. Each signal implements `async def check(decision) -> ReviewResult | None`.

```
ReviewResult:
    result: str          # success | partial | failure
    explanation: str     # human-readable reason
    confidence: float    # signal certainty (0-1)
    signal_type: str     # which signal produced this
```

| Signal | Matches When | Outcome | Confidence |
|--------|-------------|---------|------------|
| ErrorSignal | confidence < 0.4 OR description contains error/failed | failure | 0.9 |
| EpisodeSignal | Decision linked via episode_decisions | Maps episode outcome | 0.8 |
| FileExistsSignal | Description mentions file path | success if exists, skip if not | 0.7 |
| GitHubSignal | Description mentions PR # or commit hash | merged=success, closed=failure, open=skip | 0.85 |

Resolution: first result with confidence >= 0.7 wins. No match = stays unreviewed for Tier 2.

## DecisionReviewer Handler

`nous/handlers/decision_reviewer.py` — registered on `session_ended` event.

- `handle(event)`: Review session decisions, then piggyback `sweep()`.
- `sweep(max_age_days=30)`: Standalone method for all old unreviewed decisions. Ready for future scheduler.
- `_check_signals(decision)`: Iterate signals in order, return first confident match.

GitHubSignal requires `GITHUB_TOKEN` env var. Skipped gracefully if absent.

## Brain API Changes

| Method | Change |
|--------|--------|
| `brain.review()` | Add optional `reviewer: str` param |
| `brain.record()` | Accept optional `session_id: str` param |
| `brain.get_session_decisions()` | New — query by session_id |
| `brain.get_unreviewed()` | New — query unreviewed with optional stakes filter |
| `brain.generate_calibration_snapshot()` | New — compute + store snapshot |

## REST Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/decisions/{id}/review` | External review. Body: `{outcome, result, reviewer}` |
| `GET` | `/decisions/unreviewed` | Query unreviewed. Params: `?stakes=&max_age_days=&limit=` |

## Config

| Variable | Default | Description |
|----------|---------|-------------|
| `NOUS_DECISION_REVIEW_ENABLED` | `true` | Enable auto-review handler |
| `GITHUB_TOKEN` | `None` | Optional. GitHubSignal skipped if absent. |

## Phased Delivery

| Phase | What Ships | Lines (est.) |
|-------|-----------|-------------|
| Phase 1 | Migration + Brain API additions + ORM update | ~60 |
| Phase 2 | DecisionReviewer handler + 4 signals + config + wiring | ~150 |
| Phase 3 | REST endpoints + calibration snapshot generation | ~60 |
| Phase 4 | Tests | ~120 |

**Total:** ~390 lines. One new file, one migration, modifications to 6 existing files.

## Not In Scope

- Tier 3 Telegram escalation (external agent)
- Retroactive review of existing 30 decisions (Emerson bootstrap)
- Scheduler integration (sweep() is ready for it)
- Multi-reviewer consensus

## Files Changed

| File | Change |
|------|--------|
| `sql/migrations/009_decision_review.sql` | New — add session_id + reviewer columns |
| `nous/storage/models.py` | Add session_id + reviewer to Decision model |
| `nous/brain/brain.py` | Extend review/record, add get_session_decisions/get_unreviewed/generate_calibration_snapshot |
| `nous/handlers/decision_reviewer.py` | New — handler + 4 signals |
| `nous/config.py` | NOUS_DECISION_REVIEW_ENABLED, GITHUB_TOKEN |
| `nous/main.py` | Wire handler to event bus |
| `nous/cognitive/layer.py` | Pass session_id to brain.record() |
| `nous/api/rest.py` | POST /decisions/{id}/review, GET /decisions/unreviewed |
| `tests/test_decision_reviewer.py` | New — signal tests, handler tests, endpoint tests |
