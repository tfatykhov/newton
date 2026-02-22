# F007: Metrics & Growth System

**Status:** Planned
**Priority:** P1 — Measures everything, enables improvement
**Detail:** See [011-measuring-success](../research/011-measuring-success.md)

## Summary

Five-level measurement framework that tracks whether Newton is actually getting smarter, not just bigger. Fully automated via event bus. Produces weekly growth reports.

## Five Measurement Levels

### Level 1: Task Success
| Metric | How Measured | Target |
|--------|-------------|--------|
| Completion rate | Episodes with outcome=success / total | > 85% |
| User satisfaction | Explicit feedback + implicit signals | > 0.5 (scale -1 to 1) |
| Error rate | Tasks with errors / total | < 10% |

**Implicit satisfaction signals (automatic):**
- User corrections detected → negative
- User repeated request → negative
- User said thanks → positive
- User abandoned → strong negative
- Conversation flowed naturally → positive

### Level 2: Decision Quality
| Metric | How Measured | Target |
|--------|-------------|--------|
| Brier score | (confidence - outcome)² averaged | < 0.05 |
| Accuracy | Successful / reviewed decisions | > 90% |
| Confidence distribution | StdDev of confidence values | 0.08-0.20 |
| Reason diversity | % decisions with ≥2 reason types | > 70% |
| Decision velocity | Decisions per active day | 3-10 |

**Automatic outcome detection:**
- Tool success/failure → immediate
- Episode outcome → propagates to session decisions
- Superseding decisions → marks earlier as partial
- Errors → marks recent decision as failed

### Level 3: Memory Relevance
| Metric | How Measured | Target |
|--------|-------------|--------|
| Context relevance | % loaded items referenced in response | > 40% |
| Recall precision@5 | Top-5 search result relevance | > 0.7 |
| Fact accuracy | Active / (active + superseded + inactive) | > 95% |
| Memory balance | Distribution across 5 memory types | No type > 60% |

### Level 4: Growth (THE critical level)
| Metric | How Measured | Target |
|--------|-------------|--------|
| **Mistake repetition rate** | Similar failed decisions after a failure | **< 10%** |
| Censor precision | 1 - (false positives / activations) | > 85% |
| Procedure effectiveness trend | Improving vs declining procedures | More improving |
| Calibration trend | Brier score slope over snapshots | Stable or improving |

**Mistake repetition is THE metric.** If the agent never repeats mistakes, censors work, procedures improve, and calibration is honest.

### Level 5: Efficiency
| Metric | How Measured | Target |
|--------|-------------|--------|
| Tokens per task | Total tokens / completed tasks | Tracked per frame |
| Context overhead | Context tokens / total tokens per turn | < 15% |
| LLM calls per task | Total LLM calls / tasks | < 1.5 avg |
| Recall latency p95 | Context assembly time | < 200ms |

## Weekly Growth Report

Auto-generated every Sunday. Stored in `system.growth_reports`.

```
=== Newton Growth Report — Week of [date] ===

TASK SUCCESS
  Completed: X/Y (Z%)  |  Satisfaction: N  |  Errors: N

DECISIONS
  Made: N  |  Reviewed: N  |  Brier: N  |  Accuracy: N%

MEMORY
  Relevance: N%  |  Precision@5: N  |  Fact Accuracy: N%

GROWTH
  Mistake Repetition: N%  |  Censor Precision: N%
  Procedures: N improving, N stable, N declining

EFFICIENCY
  Tokens/Task: N  |  Context: N%  |  Latency p95: Nms

STRENGTHS: [auto-detected]
CONCERNS: [auto-detected]
RECOMMENDATIONS: [auto-generated]
```

## Database Tables

```sql
-- User feedback (explicit)
system.feedback (id, agent_id, episode_id, decision_id, rating, comment)

-- Per-turn context metrics (sampled)
system.context_metrics (id, agent_id, episode_id, items_loaded, items_referenced, context_tokens, total_tokens, recall_latency_ms, llm_calls)

-- Weekly reports
system.growth_reports (id, agent_id, period, report JSONB)
```

## Automation

All metrics are computed automatically via event handlers:
- Calibration snapshots: every 10 decision reviews
- Context metrics: every turn
- Growth alerts: daily tick (red flags only)
- Full growth report: weekly tick
- Memory health: daily tick
