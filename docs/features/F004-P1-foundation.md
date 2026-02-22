# F004-P1: Foundation — Postgres + Project Scaffold

**Status:** Ready to Build
**Priority:** P0 — Everything depends on this
**Estimated Effort:** 4-6 hours
**Prerequisites:** None

## Objective

Set up the Nous project foundation: Postgres container with pgvector, complete database schema (17 tables, 3 schemas), Python project scaffold with async SQLAlchemy, and basic connectivity tests.

After this phase, a developer can `docker compose up` and have a running Postgres with the full schema, and import `nous.storage` to interact with it.

## Deliverables

### 1. Docker Infrastructure

#### docker-compose.yml
```yaml
version: "3.8"

services:
  postgres:
    image: pgvector/pgvector:pg17
    container_name: nous-postgres
    environment:
      POSTGRES_DB: nous
      POSTGRES_USER: nous
      POSTGRES_PASSWORD: ${DB_PASSWORD:-nous_dev_password}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/001-init.sql
      - ./sql/seed.sql:/docker-entrypoint-initdb.d/002-seed.sql
    ports:
      - "${DB_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nous"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

#### .env.example
```
DB_PASSWORD=nous_dev_password
DB_PORT=5432
ANTHROPIC_API_KEY=your_key_here
```

### 2. Database Schema (`sql/init.sql`)

Create the complete schema from research note 008 with these specifics:

#### Extensions
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

#### Schemas
```sql
CREATE SCHEMA IF NOT EXISTS brain;
CREATE SCHEMA IF NOT EXISTS heart;
CREATE SCHEMA IF NOT EXISTS system;
```

#### Tables (create in dependency order)

**system schema (3 tables):**
1. `system.agents` — agent registry with JSONB config
2. `system.frames` — cognitive frame definitions with activation patterns
3. `system.events` — audit trail (event_type + JSONB data)

**brain schema (8 tables):**
4. `brain.decisions` — core decision records with `vector(1536)` embedding, FTS index
5. `brain.decision_tags` — normalized tags (decision_id, tag) composite PK
6. `brain.decision_reasons` — typed reasoning (analysis, pattern, empirical, etc.)
7. `brain.decision_bridge` — structure + function dual descriptions
8. `brain.thoughts` — micro-thoughts during deliberation
9. `brain.graph_edges` — decision relationships with relation type + weight
10. `brain.guardrails` — configurable rules with JSONB conditions + severity
11. `brain.calibration_snapshots` — periodic calibration metric snapshots

**heart schema (7 tables):**
12. `heart.episodes` — narrative records with embedding, tags array, outcome
13. `heart.episode_decisions` — links episodes ↔ decisions
14. `heart.procedures` — K-lines with level-bands (goals[], core_patterns[], core_tools[], core_concepts[], implementation_notes[])
15. `heart.episode_procedures` — links episodes ↔ procedures with effectiveness
16. `heart.facts` — learned information with provenance, lifecycle, superseded_by chain
17. `heart.censors` — learned constraints with auto-escalation tracking
18. `heart.working_memory` — current session state with JSONB items/threads

**Use the exact column definitions from `docs/research/008-database-design.md`**. Include ALL indexes (B-tree, ivfflat for vectors, GIN for FTS and arrays).

Note: The ivfflat indexes require data to exist. Use `CREATE INDEX ... IF NOT EXISTS` and consider deferring ivfflat index creation to a separate migration that runs after initial data load, OR use HNSW indexes instead which work on empty tables:

```sql
-- Prefer HNSW over ivfflat (works on empty tables, better recall)
CREATE INDEX idx_decisions_embedding ON brain.decisions 
    USING hnsw(embedding vector_cosine_ops);
```

### 3. Seed Data (`sql/seed.sql`)

```sql
-- Default agent
INSERT INTO system.agents (id, name, description, config) VALUES (
    'nous-default',
    'Nous',
    'A thinking agent that learns from experience',
    '{
        "identity": {"traits": ["analytical", "cautious", "curious"]},
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "working_memory_capacity": 20,
        "auto_extract_facts": true,
        "auto_create_censors": true,
        "censor_escalation": true
    }'::jsonb
);

-- Default cognitive frames
INSERT INTO system.frames (id, agent_id, name, description, activation_patterns, default_category) VALUES
    ('task', 'nous-default', 'Task Execution', 'Focused on completing a specific task', ARRAY['build', 'fix', 'create', 'implement', 'deploy'], 'tooling'),
    ('question', 'nous-default', 'Question Answering', 'Answering questions, looking things up', ARRAY['what', 'how', 'why', 'explain', 'tell me'], 'process'),
    ('decision', 'nous-default', 'Decision Making', 'Evaluating options, choosing a path', ARRAY['should', 'choose', 'decide', 'compare', 'trade-off'], 'architecture'),
    ('creative', 'nous-default', 'Creative', 'Brainstorming, ideation, exploration', ARRAY['imagine', 'brainstorm', 'what if', 'design', 'explore'], 'architecture'),
    ('conversation', 'nous-default', 'Conversation', 'Casual or social interaction', ARRAY['hello', 'hi', 'thanks', 'how are you'], 'process'),
    ('debug', 'nous-default', 'Debug', 'Investigating problems, tracing errors', ARRAY['error', 'bug', 'broken', 'failing', 'crash', 'wrong'], 'tooling');

-- Default guardrails
INSERT INTO brain.guardrails (agent_id, name, description, condition, severity) VALUES
    ('nous-default', 'no-high-stakes-low-confidence', 'Block high-stakes decisions with low confidence', '{"stakes": "high", "confidence_lt": 0.5}', 'block'),
    ('nous-default', 'no-critical-without-review', 'Block critical-stakes without explicit review', '{"stakes": "critical"}', 'block'),
    ('nous-default', 'require-reasons', 'Block decisions without at least one reason', '{"reason_count_lt": 1}', 'block'),
    ('nous-default', 'low-quality-recording', 'Block low-quality decisions (missing tags/pattern)', '{"quality_lt": 0.5}', 'block');
```

### 4. Python Project Structure

```
nous/
├── docker-compose.yml
├── Dockerfile                     # For the Nous agent (Phase 2+)
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── sql/
│   ├── init.sql                   # Full schema
│   └── seed.sql                   # Default agent, frames, guardrails
├── nous/
│   ├── __init__.py                # Version, top-level imports
│   ├── config.py                  # Settings via pydantic-settings (env vars)
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py            # Async engine, session factory, pool config
│   │   ├── models.py              # SQLAlchemy ORM models (all 17 tables)
│   │   └── migrations.py          # Schema verification / migration helper
│   ├── brain/
│   │   └── __init__.py            # Placeholder
│   ├── heart/
│   │   └── __init__.py            # Placeholder
│   ├── cognitive/
│   │   └── __init__.py            # Placeholder
│   └── api/
│       └── __init__.py            # Placeholder
├── tests/
│   ├── conftest.py                # Pytest fixtures (test DB, async session)
│   ├── test_database.py           # Connection, schema exists, CRUD basics
│   └── test_models.py             # ORM model validation
└── docs/
    ├── research/                  # (existing)
    └── features/                  # (existing)
```

#### pyproject.toml
```toml
[project]
name = "nous"
version = "0.1.0"
description = "A cognitive agent framework built on Society of Mind principles"
requires-python = ">=3.12"
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "pgvector>=0.3",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
]
```

#### nous/config.py
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    db_url: str = "postgresql+asyncpg://nous:nous_dev_password@localhost:5432/nous"
    db_pool_size: int = 10
    db_max_overflow: int = 5
    agent_id: str = "nous-default"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    log_level: str = "info"

    model_config = {"env_prefix": "NOUS_"}
```

#### nous/storage/database.py
```python
"""
Async database engine and session management.

Usage:
    db = Database(settings)
    await db.connect()
    async with db.session() as session:
        result = await session.execute(...)
    await db.disconnect()
"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from nous.config import Settings

class Database:
    def __init__(self, settings: Settings):
        self.engine = create_async_engine(
            settings.db_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            echo=settings.log_level == "debug",
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def connect(self) -> None:
        """Verify connection and schema existence."""
        async with self.engine.begin() as conn:
            # Verify schemas exist
            result = await conn.execute(
                text("SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('brain', 'heart', 'system')")
            )
            schemas = {row[0] for row in result}
            assert schemas == {"brain", "heart", "system"}, f"Missing schemas: {{'brain', 'heart', 'system'} - schemas}"

    async def disconnect(self) -> None:
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        return self.session_factory()
```

#### nous/storage/models.py

SQLAlchemy 2.0 declarative models for ALL 17+ tables. Use:
- `mapped_column()` with explicit types
- `pgvector.sqlalchemy.Vector` for embedding columns
- Proper relationship definitions with `back_populates`
- `__tablename__` and `__table_args__ = {"schema": "brain"}` etc.

#### tests/conftest.py
```python
"""
Test fixtures using a real Postgres (via docker-compose).
Tests expect `docker compose up postgres` to be running.
"""
import pytest
import pytest_asyncio
from nous.config import Settings
from nous.storage.database import Database

@pytest_asyncio.fixture
async def db():
    settings = Settings(db_url="postgresql+asyncpg://nous:nous_dev_password@localhost:5432/nous")
    database = Database(settings)
    await database.connect()
    yield database
    await database.disconnect()

@pytest_asyncio.fixture
async def session(db):
    async with db.session() as session:
        yield session
        await session.rollback()  # Clean up after each test
```

### 5. Tests

#### test_database.py
- `test_connection` — can connect to Postgres
- `test_schemas_exist` — brain, heart, system schemas present
- `test_extensions` — vector and pg_trgm extensions installed
- `test_all_tables_exist` — all 17+ tables exist with correct schemas
- `test_seed_data` — default agent, frames, guardrails exist

#### test_models.py
- `test_create_decision` — insert a decision, read it back
- `test_create_episode` — insert an episode, read it back
- `test_create_fact` — insert a fact with embedding, read it back
- `test_decision_tags` — insert decision with tags, query by tag
- `test_guardrail_check` — read guardrails, verify JSONB condition parsing

### 6. .gitignore
```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.pytest_cache/
.ruff_cache/
pgdata/
```

## Acceptance Criteria

1. `docker compose up -d postgres` starts Postgres with pgvector
2. `sql/init.sql` creates all 3 schemas, 17+ tables, all indexes
3. `sql/seed.sql` populates default agent, 6 frames, 4 guardrails
4. `nous/storage/models.py` has ORM models for all tables
5. `nous/storage/database.py` provides async connection pool
6. All tests pass: `pytest tests/ -v`
7. Clean project structure matching the tree above

## Non-Goals (This Phase)

- No REST API (Phase 2+)
- No MCP interface (Phase 2+)
- No Brain/Heart business logic (F001/F002)
- No embedding generation (needs API key, handled in F001/F002)
- No Dockerfile for Nous agent (just Postgres for now)
- No dashboard

## References

- `docs/research/008-database-design.md` — Complete SQL for all tables
- `docs/research/004-storage-architecture.md` — Design rationale
- `docs/features/F004-runtime.md` — Full runtime vision (this is Phase 1 of that)

---

*The foundation. Everything else builds on this.*
