# F008: Memory Lifecycle Management

**Status:** Planned
**Priority:** P1 — Keeps memory healthy and relevant
**Detail:** See [009-context-management](../research/009-context-management.md), [010-summarization-strategy](../research/010-summarization-strategy.md)

## Summary

Automatic lifecycle management for all memory types. Memories aren't static — they confirm, age, consolidate, and retire. Handled by the Memory Maintainer via daily tick events.

## Lifecycles

### Facts
```
NEW ──→ CONFIRMED ──→ CORE ──→ SUPERSEDED ──→ INACTIVE
 │         │            │          │
 │     Re-encountered  3+ confirms  Contradicted
 │     last_confirmed  Higher rank   by newer fact
 │     updated         in search     
 │                                 
 Proven wrong ──────────────────→ INACTIVE (active=false)
```

**Automatic transitions:**
- NEW → CONFIRMED: Same fact encountered again (semantic match > 0.95)
- CONFIRMED → CORE: confirmation_count ≥ 3
- Any → SUPERSEDED: New contradicting fact stored (superseded_by set)
- Any → INACTIVE: Proven false (active = false)

**Deduplication on learn():** Before storing, search for similar facts. Score > 0.95 = confirm existing. Score 0.85-0.95 = LLM checks if semantically identical.

### Episodes
```
ACTIVE ──→ SUMMARIZED ──→ ARCHIVED
(< 30 days)  (30-90 days)   (> 90 days)

ACTIVE: Full detail available. Raw transcript stored.
SUMMARIZED: Detail trimmed to first 2,000 chars. Summary preserved.
ARCHIVED: Detail removed entirely. Only summary + metadata + extracted facts remain.
```

**Automatic transitions (daily tick):**
- 30 days: Trim detail, ensure summary exists, ensure facts extracted
- 90 days: Remove detail entirely

**On episode close (automatic):**
1. LLM generates title + summary (100-150 words)
2. Facts extracted from summary
3. Decisions linked
4. Outcome assessed (explicit or implicit)

### Procedures (K-Lines)
```
ACTIVE ──→ EFFECTIVE ──→ (keeps improving)
   │                         
   └──→ INEFFECTIVE ──→ RETIRED
        (< 40% after     (active=false)
         5+ activations)
```

**Automatic tracking:**
- Every activation: activation_count++
- Episode with procedure succeeded: success_count++
- Episode with procedure failed: failure_count++
- Effectiveness = success_count / activation_count (auto-computed)
- Flagged for review when effectiveness < 0.4 after 5+ uses

**Compression (daily tick):** When implementation_notes > 10, LLM consolidates to ≤ 5 essential notes.

### Censors
```
CREATED ──→ ACTIVE ──→ ESCALATED ──→ (permanent or retired)
(warn)      (triggering)  (block)
                              │
                              └──→ RETIRED (too many false positives)
```

**Automatic transitions:**
- Created from failed decisions (severity=warn, created_by=auto_failure)
- Escalated when activation_count ≥ escalation_threshold (warn → block)
- Retired when false_positive_rate > 50% after 5+ activations

### Decisions
```
RECORDED ──→ PENDING ──→ REVIEWED
(intent)     (work done)   (outcome logged)
```

No deletion. No archiving. Decisions are permanent. Even failures are valuable training data for calibration.

**Automatic outcome detection:**
- Tool success/failure → immediate review
- Episode outcome → propagates to session decisions
- Error during session → marks most recent decision as failed
- Superseding decision detected → marks earlier as partial

## Fact Generalization (Compaction)

*Added from LangChain Agent Builder memory lessons (research/013)*

**Problem:** Agents accumulate specific facts instead of generalizing. Example: 15 facts about specific vendors to ignore, instead of one rule "ignore cold outreach."

**Solution:** When fact count in a domain exceeds threshold, trigger generalization:

```
Event: fact_threshold_exceeded (domain has > 10 active facts)
  ↓
Handler: generalize_facts
  1. Load all active facts in domain
  2. LLM call: "Merge these N specific facts into 1-3 general rules"
  3. Store general rules as new CORE facts
  4. Mark originals as SUPERSEDED (superseded_by = general fact ID)
  5. Log compaction event for metrics
```

**Trigger:** Event Bus handler (F006) on `fact_learned` when domain count > threshold.

**Validation on write:** Heart.facts.learn() should also check for contradictions:
- Embedding similarity > 0.9 with existing fact but different content → flag for resolution
- Surface contradictions back to Cognitive Layer rather than silently storing both

## Memory Volume Management

Estimated per agent per year: ~80 MB. Postgres handles this trivially.

But for very long-lived agents (years of operation), additional management:
- Facts: dedup prevents unbounded growth
- Episodes: archiving keeps only summaries (biggest compression)
- Procedures: rarely grow beyond ~500
- Censors: retirement prevents accumulation
- Decisions: permanent but compact (~2KB each)

## Maintenance Schedule

| Action | Trigger | Frequency |
|--------|---------|-----------|
| Episode trimming | daily_tick | Daily |
| Episode archiving | daily_tick | Daily |
| Superseded fact deactivation | daily_tick | Daily |
| Ineffective procedure flagging | daily_tick | Daily |
| Noisy censor retirement | daily_tick | Daily |
| Procedure compression | daily_tick | Daily (when notes > 10) |
| Fact deduplication | fact_learned event | Every new fact |
| Censor escalation | censor_triggered event | Every trigger |
| Calibration snapshot | Every 10 reviews | Continuous |
