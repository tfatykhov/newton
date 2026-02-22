# F005: Context Engine

**Status:** Planned
**Priority:** P0 — Core capability
**Detail:** See [009-context-management](../research/009-context-management.md), [010-summarization-strategy](../research/010-summarization-strategy.md)

## Summary

The Context Engine assembles the right information at the right detail level for each interaction. It's the bridge between Heart/Brain's stored memories and the LLM's context window.

## Key Design

### Token Budget (Frame-Adaptive)

| Frame | Budget | Rationale |
|-------|--------|-----------|
| conversation | 3K | Lightweight — identity + facts |
| question | 6K | Needs facts + episodes |
| task | 8K | Needs procedures + facts |
| decision | 12K | Full picture — decisions, procedures, episodes, facts |
| debug | 10K | Procedures + episodes + error patterns |

### Eight Priority Layers

| Layer | Always? | Budget | Content |
|-------|---------|--------|---------|
| 1. Identity | ✅ | 500 | Agent name, traits, core rules |
| 2. Censors | ✅ | 300 | Active constraints |
| 3. Frame | ✅ | 500 | Current cognitive frame + prompts |
| 4. Working Memory | ✅ | 700 | Current task, open threads |
| 5. Decisions | Selected | 2000 | Similar past decisions (summaries) |
| 6. Facts | Selected | 1500 | Known information about topic |
| 7. Procedures | Selected | 1500 | How-to knowledge (core band only) |
| 8. Episodes | Selected | 1000 | Past experiences (summaries) |

### Three-Tier Summarization

| Tier | When | Cost | Method |
|------|------|------|--------|
| Pre-computed | Write time | Free | Templates for decisions/facts; LLM for episodes |
| Query-relevant | Context assembly | Cached LLM (rare) | Extract relevant portion for complex records |
| Full expansion | On explicit request | DB read | Only when agent asks "tell me more" |

### Relevance Scoring

Composite score for ranking search results:
- 50% semantic similarity
- 15% frame type priority
- 15% recency (30-day half-life)
- 10% outcome quality (successful > failed)
- 5% usage frequency
- 5% confidence level

### Context Refresh

Triggers for rebuilding context mid-conversation:
- Frame switch detected
- Topic drifts from current context (similarity < 0.5)
- Decision point reached (pre-action)
- Token pressure in long conversations

## Interface

```python
from nous.cognitive import ContextEngine

ctx = ContextEngine(brain, heart, embedder)

# Build context for a new input
context_str = await ctx.build(
    input="Should we use Redis for caching?",
    frame=decision_frame,
    agent_id="nous-1"
)

# Check if refresh needed on follow-up
refresh = await ctx.refresh_if_needed("What about Memcached instead?")

# Expand a specific memory for detail
detail = await ctx.expand(memory_id="abc-123")
```

## Metrics

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| Context relevance | > 40% items referenced | Are we loading useful stuff? |
| Budget utilization | 50-90% | Using space wisely? |
| Context refresh rate | < 0.3 per turn | Not rebuilding too often? |
| Cold start time | < 500ms | Fast enough? |
| Context overhead | < 15% of total tokens | Not dominating the conversation? |
