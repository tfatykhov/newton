# Nous Feature Index

## v0.1.0 — "The Thinking Agent"

### P0: Core Architecture
| Feature | Name | Status | Description |
|---------|------|--------|-------------|
| F001 | [Brain Module](F001-brain-module.md) | ✅ Shipped | Decision intelligence — recording, deliberation, calibration, guardrails, graph |
| F002 | [Heart Module](F002-heart-module.md) | ✅ Shipped | Memory system — episodic, semantic, procedural, working, censors |
| F003 | [Cognitive Layer](F003-cognitive-layer.md) | ✅ Shipped | The Nous Loop — frames, recall, deliberation, monitoring, end-of-session reflection |
| F004 | [Runtime](F004-runtime.md) | ✅ Shipped | Docker container, REST API, MCP interface, Telegram bot |
| F005 | [Context Engine](F005-context-engine.md) | ✅ Shipped | Frame-adaptive context assembly (implemented as `cognitive/context.py`) |
| F006 | [Event Bus](F006-event-bus.md) | ✅ Shipped | In-process async event bus, automated handlers |

### P1: Intelligence & Measurement
| Feature | Name | Status | Description |
|---------|------|--------|-------------|
| F007 | [Metrics & Growth](F007-metrics-growth.md) | Planned | 5-level measurement framework, weekly growth reports, automatic tracking |
| F008 | [Memory Lifecycle](F008-memory-lifecycle.md) | Planned | Auto lifecycle for all memory types — confirm, trim, archive, escalate, retire, **generalize** |

### P1: Capabilities
| Feature | Name | Status | Description |
|---------|------|--------|-------------|
| F009 | [Async Subtasks](F009-async-subtasks.md) | Planned | Background task queue — parallel execution, non-blocking chat, Postgres-backed workers |
| F010 | [Memory Improvements](F010-memory-improvements.md) | ✅ Shipped | Episode summaries, clean decision descriptions, proactive fact learning, user-tagged episodes |

### Implementation Specs

All shipped implementation specs with PR references:

| Spec | Name | Status | PR |
|------|------|--------|-----|
| 001 | Postgres Scaffold | ✅ Shipped | #1 |
| 002 | Brain Module | ✅ Shipped | #2 |
| 003 | Heart Module | ✅ Shipped | #3 |
| 003.1 | Heart Enhancements | ✅ Shipped | #6 |
| 003.2 | Frame-Tagged Encoding | ✅ Shipped | — |
| 004 | Cognitive Layer | ✅ Shipped | #10 |
| 004.1 | CEL Guardrails | ✅ Shipped | #10 |
| 005 | Runtime (REST + MCP + Runner) | ✅ Shipped | — |
| 005.1 | Smart Context Preparation | ✅ Shipped | — |
| 005.2 | Direct API Rewrite | ✅ Shipped | #15 |
| 005.3 | Web Tools | ✅ Shipped | #16 |
| 005.4 | Streaming Responses | ✅ Shipped | #23 |
| 005.5 | Noise Reduction | ✅ Shipped | #20 |
| 006 | Event Bus | ✅ Shipped | — |
| 006.2 | Context Quality | ✅ Shipped | #31 |
| 007 | Extended Thinking | ✅ Shipped | — |
| 007.1 | Thinking Indicators | ✅ Shipped | #53 |
| 007.2 | Topic-Aware Recall | ✅ Shipped | #55 |
| 007.3 | Improve _is_informational() | ✅ Shipped | #55 |
| 007.4 | Fix Unpopulated Columns | ✅ Shipped | #55 |
| 007.5 | Recall Min Threshold | ✅ Shipped | #59 |

### v0.2.0 Preview (Future)
| Feature | Name | Description |
|---------|------|-------------|
| F011 | [Skill Discovery](F011-skill-discovery.md) | Index workspace skills as procedures, auto-surface in RECALL stage based on task/frame |
| F012 | K-Line Learning | Auto-create/refine procedures from repeated patterns |
| F013 | Frame Splitting | Parallel cognitive frames via sub-agents |
| F014 | Model Router | LLM portability via proxy layer |
| F015 | Growth Engine | Administrative self-improvement (Papert's Principle) |
| F016 | Multi-Agent | Nous agents sharing knowledge |
| F017 | Dashboard | Visual growth tracking and cognitive state |
| F018 | [Agent Identity](F018-agent-identity.md) | DB-backed identity layer — character, values, protocols that evolve over time |

## Stats

- **Total source:** ~11,800 lines of Python
- **Test count:** 424 tests across 33 test files
- **Database:** 18 tables across 3 schemas (brain, heart, system)
- **Tools:** 8 agent tools (record_decision, recall_deep, learn_fact, create_censor, bash, read_file, write_file, web_search, web_fetch)
- **Endpoints:** 12 REST endpoints + MCP server + Telegram bot

## Research Notes

| # | Title | Key Topic |
|---|-------|-----------|
| [001](../research/001-foundations.md) | Foundations | Problem statement, Nous hypothesis |
| [002](../research/002-minsky-mapping.md) | Minsky Mapping | 14 chapters → Nous components |
| [003](../research/003-runtime-decision.md) | Runtime Decision | Claude Agent SDK + model router |
| [004](../research/004-storage-architecture.md) | Storage Architecture | Postgres + pgvector, swappable backends |
| [005](../research/005-cognitive-layer.md) | Cognitive Layer | The seven systems |
| [006](../research/006-v01-features.md) | v0.1.0 Features | Initial feature plan |
| [007](../research/007-memory-integration.md) | Memory Integration | 5 memory types, CE integration |
| [008](../research/008-database-design.md) | Database Design | 20 tables, 3 schemas, full SQL |
| [009](../research/009-context-management.md) | Context Management | Token budgets, relevance scoring |
| [010](../research/010-summarization-strategy.md) | Summarization | 3-tier compression, episode lifecycle |
| [011](../research/011-measuring-success.md) | Measuring Success | 5-level metrics, growth reports |
| [012](../research/012-automation-pipeline.md) | Automation Pipeline | Event bus, 7 handlers, full wiring |
| [013](../research/013-langchain-memory-lessons.md) | LangChain Memory Lessons | 5 takeaways: reflection, generalization, validation, approval gates |
| [014](../research/014-group-evolving-agents.md) | GEA | Experience sharing for open-ended self-improvement |
| [015](../research/015-deep-thinking-ratio.md) | DTR | Measuring real reasoning effort, not token count |

## Architecture Summary

![Nous Architecture](../nous-architecture.png)

## Database: 18 Tables, 3 Schemas

| Schema | Tables | Purpose |
|--------|--------|---------|
| `brain` (7) | decisions, decision_tags, decision_reasons, decision_bridge, thoughts, graph_edges, guardrails, calibration_snapshots | Decision intelligence |
| `heart` (7) | episodes, episode_decisions, episode_procedures, facts, procedures, censors, working_memory | Memory system |
| `system` (4) | agents, frames, events, context_metrics | Config, tracking |
