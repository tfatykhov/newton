# Nous Feature Index

## v0.1.0 — "The Thinking Agent"

### P0: Core Architecture
| Feature | Name | Status | Description |
|---------|------|--------|-------------|
| F001 | [Brain Module](F001-brain-module.md) | Shipped | Decision intelligence — recording, deliberation, calibration, guardrails, graph |
| F002 | [Heart Module](F002-heart-module.md) | Shipped | Memory system — episodic, semantic, procedural, working, censors |
| F003 | [Cognitive Layer](F003-cognitive-layer.md) | Spec Ready | The Nous Loop — frames, recall, deliberation, monitoring, end-of-session reflection |
| F004 | [Runtime](F004-runtime.md) | Spec Ready | Docker container, REST API, MCP interface, project structure |
| F005 | [Context Engine](F005-context-engine.md) | Merged into F003 | Frame-adaptive context assembly (implemented as `cognitive/context.py`) |
| F006 | [Event Bus](F006-event-bus.md) | Planned | In-process async event bus, 7 automated handlers, 27/29 actions automatic |

### P1: Intelligence & Measurement
| Feature | Name | Status | Description |
|---------|------|--------|-------------|
| F007 | [Metrics & Growth](F007-metrics-growth.md) | Planned | 5-level measurement framework, weekly growth reports, automatic tracking |
| F008 | [Memory Lifecycle](F008-memory-lifecycle.md) | Planned | Auto lifecycle for all memory types — confirm, trim, archive, escalate, retire, **generalize** |

### P1: Capabilities
| Feature | Name | Status | Description |
|---------|------|--------|-------------|
| F009 | [Async Subtasks](F009-async-subtasks.md) | Planned | Background task queue — parallel execution, non-blocking chat, Postgres-backed workers |

### v0.2.0 Preview (Future)
| Feature | Name | Description |
|---------|------|-------------|
| F010 | K-Line Learning | Auto-create/refine procedures from repeated patterns |
| F011 | Frame Splitting | Parallel cognitive frames via sub-agents |
| F012 | Model Router | LLM portability via proxy layer |
| F013 | Growth Engine | Administrative self-improvement (Papert's Principle) |
| F014 | Multi-Agent | Nous agents sharing knowledge |
| F015 | Dashboard | Visual growth tracking and cognitive state |

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

## Architecture Summary

```
┌──────────────────────────────────────────────┐
│              Nous Agent (F004)              │
│                                               │
│  ┌──────────────────────────────────────┐    │
│  │      Cognitive Layer (F003)           │    │
│  │  Frame → Recall → Deliberate →        │    │
│  │  → Act → Monitor → Learn              │    │
│  └────────┬─────────────────┬────────────┘    │
│           │                 │                  │
│  ┌────────▼──────┐ ┌───────▼─────────┐       │
│  │  Brain (F001)  │ │  Heart (F002)    │       │
│  │  Decisions     │ │  Episodes        │       │
│  │  Deliberation  │ │  Facts           │       │
│  │  Calibration   │ │  Procedures      │       │
│  │  Guardrails    │ │  Censors         │       │
│  │  Graph         │ │  Working Memory  │       │
│  └────────┬──────┘ └───────┬─────────┘       │
│           │                 │                  │
│  ┌────────▼─────────────────▼──────────┐      │
│  │  Context Engine (F005)               │      │
│  │  Token budgets, relevance scoring,   │      │
│  │  3-tier summarization                │      │
│  └──────────────────────────────────────┘      │
│                                               │
│  ┌──────────────────────────────────────┐     │
│  │  Event Bus (F006)                     │     │
│  │  7 handlers, 27/29 actions automatic  │     │
│  └──────────────────────────────────────┘     │
│                                               │
│  ┌──────────────────────────────────────┐     │
│  │  Metrics & Growth (F007)              │     │
│  │  5 levels, weekly reports             │     │
│  └──────────────────────────────────────┘     │
│                                               │
│  ┌──────────────────────────────────────┐     │
│  │  Memory Lifecycle (F008)              │     │
│  │  Auto confirm/trim/archive/retire     │     │
│  └──────────────────────────────────────┘     │
│                                               │
│  ┌──────────────┐  ┌────────────────────┐     │
│  │ REST API      │  │ MCP (external)     │     │
│  └──────────────┘  └────────────────────┘     │
└───────────────────────┬───────────────────────┘
                        │
                   ┌────▼─────┐
                   │ Postgres  │
                   │ pgvector  │
                   │ 20 tables │
                   └──────────┘
```

## Database: 20 Tables, 3 Schemas

| Schema | Tables | Purpose |
|--------|--------|---------|
| `brain` (8) | decisions, decision_tags, decision_reasons, decision_bridge, thoughts, graph_edges, guardrails, calibration_snapshots | Decision intelligence |
| `heart` (7) | episodes, episode_decisions, episode_procedures, facts, procedures, censors, working_memory | Memory system |
| `system` (5) | agents, frames, events, feedback, context_metrics, growth_reports | Config, tracking, metrics |
