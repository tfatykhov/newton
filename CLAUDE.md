# CLAUDE.md - Nous Development Guide

## What is Nous?

Nous (Greek: mind/intellect) is a cognitive agent framework built on Minsky's Society of Mind principles. It gives AI agents persistent memory, decision intelligence, and the ability to learn from experience.

## Architecture

```
Cognitive Layer (hooks into LLM calls)
    ├── Brain (decisions, deliberation, calibration, guardrails)
    ├── Heart (episodes, facts, procedures, censors, working memory)
    ├── Context Engine (token budgets, relevance scoring, summarization)
    └── Event Bus (async handlers for automation)

Storage: PostgreSQL + pgvector (one DB, three schemas: brain/heart/system)
API: REST + MCP (external interface only — internal calls are Python)
```

## Project Structure

```
nous/
├── docker-compose.yml          # Postgres + pgvector
├── sql/
│   ├── init.sql                # Full schema (17 tables, 3 schemas)
│   └── seed.sql                # Default agent, frames, guardrails
├── nous/                       # Python package
│   ├── config.py               # Settings via pydantic-settings
│   ├── storage/                # Database layer (async SQLAlchemy)
│   ├── brain/                  # Decision intelligence organ
│   ├── heart/                  # Memory system organ
│   ├── cognitive/              # Cognitive layer (Nous Loop)
│   └── api/                    # REST + MCP endpoints
├── tests/                      # pytest + pytest-asyncio
└── docs/
    ├── research/               # Theory & design notes (001-012)
    ├── features/               # High-level feature specs (F001-F014)
    └── implementation/         # Build-ready specs (001, 002, ...)
```

## How to Work

### Read Before Building

1. Check `docs/implementation/` for the current build spec - these are your instructions
2. Reference `docs/research/` for design rationale (especially `008-database-design.md` for schema details)
3. Reference `docs/features/` for high-level feature context

### Tech Stack

- **Python 3.12+**
- **PostgreSQL 17** with pgvector extension
- **SQLAlchemy 2.0+** (async, declarative ORM)
- **asyncpg** (async Postgres driver)
- **pydantic v2** + pydantic-settings for config
- **pytest** + pytest-asyncio for tests

### Key Principles

- **Brain and Heart are in-process Python modules** - no MCP, no HTTP between them. Direct function calls, shared connection pool.
- **MCP is only the external interface** - for other agents/tools to talk to Nous.
- **Same ideas as Cognition Engines, not same code** - CE proved the concepts, Nous reimplements natively.
- **Async everywhere** - all database operations use async/await.
- **pgvector for all embeddings** - unified semantic search, no separate vector DB.
- **HNSW indexes over ivfflat** - works on empty tables, better recall.

### Database

- Three schemas: `brain`, `heart`, `system`
- All tables are agent-scoped (`agent_id` column) for multi-agent readiness
- Use `vector(1536)` for embeddings (text-embedding-3-small)
- Full-text search via `tsvector` + GIN indexes
- JSONB for flexible fields (config, conditions, items)
- Soft deletes (`active` boolean), never hard delete memory

### Code Style

- Type hints on everything
- Docstrings on public functions
- Use `mapped_column()` for SQLAlchemy models
- Use `pydantic.BaseModel` for API schemas
- Async context managers for database sessions
- Tests use real Postgres (via docker-compose), not mocks

### Running

```bash
# Start Postgres
docker compose up -d postgres

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start Nous (when ready)
python -m nous.main
```

### Environment Variables

All prefixed with `NOUS_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NOUS_DB_URL` | `postgresql+asyncpg://nous:nous_dev_password@localhost:5432/nous` | Database connection |
| `NOUS_DB_POOL_SIZE` | `10` | Connection pool size |
| `NOUS_AGENT_ID` | `nous-default` | Agent identifier |
| `NOUS_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `NOUS_LOG_LEVEL` | `info` | Log level |

## Implementation Specs

Build specs live in `docs/implementation/` and are numbered sequentially:

| # | Name | Description | Status |
|---|------|-------------|--------|
| 001 | [Postgres Scaffold](docs/implementation/001-postgres-scaffold.md) | DB schema + Python project setup | Ready |

**Each spec contains:**
- Exact deliverables with code snippets
- Acceptance criteria (pass/fail)
- Test requirements
- Non-goals (what NOT to build)
- References to research/feature docs

**Follow the spec.** If something is ambiguous, check the referenced research notes. If still unclear, ask.

## Git Workflow

- Work on feature branches, not main
- Commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Keep commits focused - one logical change per commit

## References

- [Society of Mind](docs/research/002-minsky-mapping.md) - How Minsky maps to Nous
- [Database Design](docs/research/008-database-design.md) - Complete SQL for all 17 tables
- [Storage Architecture](docs/research/004-storage-architecture.md) - Why Postgres + pgvector
- [Cognitive Layer](docs/research/005-cognitive-layer.md) - The seven systems
- [Automation Pipeline](docs/research/012-automation-pipeline.md) - Event bus design
