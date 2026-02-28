# Design: 008.4 Episode Summary Quality Improvements

**Date:** 2026-02-28
**Spec:** docs/implementation/008.4-summary-quality.md
**Depends on:** 008.3 (shipped, PR #79)

## Overview

Improve episode summarizer output quality across 4 dimensions: outcome classification, decision awareness, lesson quality, and transcript truncation. Coordinate fact extraction to eliminate redundant LLM calls.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | All 4 spec changes | Full quality improvement in one push |
| Delivery | 2 PRs | PR 1: prompt + facts (highest impact, lowest risk). PR 2: brain context + truncation (structural change). |
| Brain access | Constructor injection | EpisodeSummarizer receives Brain directly. Explicit, follows existing Heart pattern. |
| Fact strategy | Exclusive with fallback | candidate_facts from summarizer stored directly. Skip fact extractor LLM call. Fall back to current extraction when no candidates. |
| outcome_rationale storage | JSONB only | Stored in structured_summary blob. No schema change. Promote to column later if F007 needs it. |
| Truncation | Spec's priority-based approach | Keyword heuristics to score turns. Keeps first/last, fills middle by score. |
| Tests | Unit + integration | Unit tests for _truncate_transcript, _build_decision_context. Integration tests for full handler flow. |

## PR 1: Enhanced Prompt + Fact Coordination

### Files Changed

| File | Change |
|------|--------|
| `nous/handlers/episode_summarizer.py` | Replace `_SUMMARY_PROMPT` with structured prompt |
| `nous/handlers/fact_extractor.py` | Accept `candidate_facts` from event, skip LLM when present |
| `tests/test_episode_summarizer.py` | Unit tests for new prompt fields |
| `tests/test_event_bus.py` | Integration test for candidate_facts passthrough |

### Enhanced Prompt

Replace `_SUMMARY_PROMPT` with:
- **Outcome guidelines**: Clear definitions of resolved/partial/unresolved/informational
- **`outcome_rationale`**: Forces LLM to reason about classification (stored in JSONB)
- **Lesson-focused `key_points`**: "What to remember next time" not "what happened"
- **`candidate_facts`**: Factual statements worth storing as standalone facts

New structured summary schema:
```json
{
  "title": "<5-10 word descriptive title>",
  "summary": "<100-150 word prose summary>",
  "key_points": ["<reusable lesson>", "<pattern or insight>"],
  "outcome": "<resolved|partial|unresolved|informational>",
  "outcome_rationale": "<1 sentence explaining classification>",
  "topics": ["<topic1>", "<topic2>"],
  "candidate_facts": ["<factual statement worth storing>"]
}
```

### Fact Extractor Change

When `candidate_facts` present in `episode_summarized` event data:
1. Store each candidate directly via `heart.learn(FactInput(...))` with existing dedup
2. Skip LLM extraction call entirely

Fallback: When no `candidate_facts` in event data, use current LLM extraction (backward compatibility).

### Tests (PR 1)

1. **test_new_prompt_fields_parsed** - Verify outcome_rationale and candidate_facts extracted from LLM response
2. **test_candidate_facts_stored_directly** - Fact extractor uses candidates, skips LLM
3. **test_fact_extractor_fallback** - Falls back to LLM when no candidates
4. **test_candidate_facts_dedup** - Duplicate candidates filtered by existing dedup

## PR 2: Brain Context + Smart Truncation

### Files Changed

| File | Change |
|------|--------|
| `nous/handlers/episode_summarizer.py` | Add `_build_decision_context()`, `_truncate_transcript()` |
| `nous/main.py` | Pass `brain` to EpisodeSummarizer constructor |
| `tests/test_episode_summarizer.py` | Unit tests for new methods |
| `tests/test_event_bus.py` | Integration test with Brain dependency |

### Decision Context Injection

New constructor signature:
```python
EpisodeSummarizer(heart, brain, settings, bus, http)
```

New method `_build_decision_context(episode_id)`:
- Fetches episode detail -> gets `decision_ids`
- For each decision ID, fetches decision detail from Brain
- Returns formatted string: "Decisions made during this episode: ..."
- Try/except -> returns empty string on any failure

Injected into prompt via `{decision_context}` placeholder.

### Smart Transcript Truncation

New method `_truncate_transcript(transcript, max_chars=8000)`:
- Split transcript into turns on `\n\n`
- Score each turn:
  - +2.0: decision language ("decided", "chose", "because", "learned", "conclusion")
  - +1.0: user turns (`user:` prefix)
  - -1.0: long tool outputs (>500 chars with code blocks or many newlines)
- Always keep first and last turns
- Fill middle by score within budget
- Reconstruct in original order

Replaces current crude `[:3800] + [-3800:]` approach.

### Tests (PR 2)

1. **test_truncate_preserves_first_last** - First and last turns always kept
2. **test_truncate_prioritizes_decisions** - Decision turns scored higher, kept over tool output
3. **test_truncate_noop_under_limit** - Short transcript returned unchanged
4. **test_build_decision_context** - Returns formatted string with linked decisions
5. **test_build_decision_context_no_decisions** - Returns empty string when none linked
6. **test_build_decision_context_error_handling** - Returns empty string on Brain errors
7. **test_full_handler_with_brain** - Integration: enriched summary includes decision context

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Outcome = "partial" rate | ~90% | < 40% |
| Episodes with useful lessons | ~20% | > 70% |
| Fact source tracing | 0% | > 50% (via candidate_facts) |
| LLM calls per episode close | 2 (summarizer + fact extractor) | 1 (summarizer only, when candidates present) |

## Risks

| Risk | Mitigation |
|------|------------|
| Longer prompt = more tokens | Background model is cheap. ~200 extra tokens/episode. |
| Decision context fetch adds latency | Async, try/except with empty string fallback. |
| candidate_facts may duplicate existing facts | Existing dedup (cosine > 0.85) handles this. |
| Truncation heuristics may cut wrong things | Conservative: user turns and decision language always kept. First/last always preserved. |
| Brain constructor dependency | Falls back gracefully. Summarizer works without Brain (empty decision context). |
