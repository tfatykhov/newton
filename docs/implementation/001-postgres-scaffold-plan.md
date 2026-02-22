# 001: Postgres Scaffold — Implementation Plan

**Source Spec:** [001-postgres-scaffold.md](001-postgres-scaffold.md)
**Reviewed By:** nous-arch (architecture), nous-db (database design), nous-devil (devil's advocate)
**Synthesized By:** forge-lead | **Date:** 2026-02-22
**Decision IDs:** `8cd0dabf` (team), `b0d60555` (synthesis), `0dcbe9a1` (arch), `9e6fff70` (db), `a1d89a31` (devil)

---

## Resolved Design Decisions

These decisions resolve ambiguities and contradictions found across the spec and research docs. All three reviewers were consulted; unanimous items are marked.

| # | Decision | Rationale | Reviewers |
|---|----------|-----------|-----------|
| D1 | **HNSW on all vector indexes** — replace every `ivfflat` from research/008 | ivfflat fails on empty tables (returns 0 results). HNSW works immediately, better recall (~99% vs ~95%), acceptable memory at our scale (~300MB for 10K vectors across 5 tables). Default params (m=16, ef_construction=64) are fine for Phase 1. | All 3 (unanimous) |
| D2 | **`raise RuntimeError` instead of `assert`** in `Database.connect()` | Python `-O` strips asserts. Schema validation must never be silently skipped. | All 3 (unanimous) |
| D3 | **SAVEPOINT pattern for test fixtures** — not plain rollback | Plain `session.rollback()` is a no-op after `session.commit()`. SAVEPOINT wraps the entire test so even committed data is rolled back. | arch + devil |
| D4 | **Defer `search_hybrid()` function** — do NOT include in init.sql | References nonexistent `content_text` column. Dynamic SQL has injection risk. Search logic belongs in Python (Phase 2+). | All 3 (unanimous) |
| D5 | **Guardrails: `UNIQUE(agent_id, name)`** not `UNIQUE(name)` | Global uniqueness on name breaks multi-agent. Two agents must be able to have same-named guardrails. | db |
| D6 | **Add `updated_at` trigger function** | Without a trigger, `updated_at` stays at creation time forever. 5 lines of SQL, prevents silent stale timestamps. | db + devil |
| D7 | **Add `CHECK` constraints on enum-like VARCHAR columns** | Prevents typos at the DB level. Minimal overhead, good DX. | db |
| D8 | **Use generated `tsvector` columns for FTS** (not expression indexes) | Expression-based GIN indexes require repeating the exact expression in every query. Generated columns (`GENERATED ALWAYS AS ... STORED`) are cleaner for ORM and avoid duplication. | arch |
| D9 | **Build ORM models for ALL 18 tables** | Mechanical work but ensures schema-ORM alignment upfront. Cheaper than retrofitting later. Models for unused tables serve as documentation. | forge-lead (overriding devil's cut recommendation) |
| D10 | **Cut `migrations.py`** from deliverables | Undefined in spec, no acceptance criteria. Schema verification is in `Database.connect()`. Real migrations use Alembic (future). | arch + devil |
| D11 | **Table count is 18, not 17** | brain=8, heart=7, nous_system=3. Research doc 008 summary miscounts brain as 7. | arch + db |
| D12 | **Rename `system` schema to `nous_system`** | `SYSTEM` is non-reserved in PG (works unquoted) but causes practical friction — ORMs, GUI tools (pgAdmin, DBeaver), Alembic, and raw SQL may require quoting or misinterpret it. Rename cost is zero now (no code exists), but high later (every FK, model, test, query). `nous_system` is clear and project-scoped. | user review (overriding db's "safe" assessment) |
| D13 | **`db` fixture should be session-scoped** | Creating a new engine + pool per test is slow. Session-scoped `db` with function-scoped `session` gives isolation without overhead. | arch |
| D14 | **Add composite index `decisions(agent_id, created_at DESC)`** | Most common query: "my recent decisions". Separate indexes on each column won't combine efficiently. | db |
| D15 | **Add missing index on `frames(agent_id)`** | PG does not auto-create FK indexes. Queries filtering frames by agent will seq scan without this. | db |
| D16 | **Remove `version: "3.8"` from docker-compose.yml** | Deprecated in Compose V2, produces warning. | arch + db |

---

## Execution Order

Build in this order. Each step's dependencies are listed.

### Phase A: Infrastructure (no code dependencies)

| Step | File | Description | Depends On |
|------|------|-------------|------------|
| A1 | `docker-compose.yml` | Postgres container with pgvector. Remove `version` key. | Nothing |
| A2 | `.env.example` | Environment variable template | Nothing |
| A3 | `.gitignore` | Python + Docker + IDE ignores | Nothing |
| A4 | `pyproject.toml` | Python project with pinned dependency ranges | Nothing |

### Phase B: Database Schema (depends on A1)

| Step | File | Description | Depends On |
|------|------|-------------|------------|
| B1 | `sql/init.sql` | Extensions, schemas, trigger function, all 18 tables, all indexes | A1 |
| B2 | `sql/seed.sql` | Default agent, 6 frames, 4 guardrails | B1 |

### Phase C: Python Package (depends on A4)

| Step | File | Description | Depends On |
|------|------|-------------|------------|
| C1 | `nous/__init__.py` | Version, top-level exports | A4 |
| C2 | `nous/config.py` | Settings via pydantic-settings | C1 |
| C3 | `nous/storage/__init__.py` | Storage module exports | C1 |
| C4 | `nous/storage/database.py` | Async engine, session factory, connection verification | C2, C3 |
| C5 | `nous/storage/models.py` | All 18 SQLAlchemy ORM models | C3 |
| C6 | `nous/brain/__init__.py` | Placeholder | C1 |
| C7 | `nous/heart/__init__.py` | Placeholder | C1 |
| C8 | `nous/cognitive/__init__.py` | Placeholder | C1 |
| C9 | `nous/api/__init__.py` | Placeholder | C1 |

### Phase D: Tests (depends on B2, C5)

| Step | File | Description | Depends On |
|------|------|-------------|------------|
| D1 | `tests/conftest.py` | Async fixtures with SAVEPOINT isolation | C4 |
| D2 | `tests/test_database.py` | Connection, schemas, extensions, tables, seed data | D1, B2 |
| D3 | `tests/test_models.py` | CRUD operations via ORM models | D1, C5 |

---

## File-by-File Implementation Notes

### `docker-compose.yml`

```yaml
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

**Notes:**
- No `version` key (D16).
- After modifying init.sql/seed.sql, you MUST run `docker compose down -v && docker compose up -d` — init scripts only execute on first volume creation.

### `sql/init.sql`

**Structure** (in order):
1. Extensions (`vector`, `pg_trgm`)
2. Schemas (`brain`, `heart`, `nous_system`)
3. Trigger function `update_timestamp()` (D6)
4. `nous_system` tables: agents, frames, events
5. `brain` tables: decisions, decision_tags, decision_reasons, decision_bridge, thoughts, graph_edges, guardrails, calibration_snapshots
6. `heart` tables: episodes, episode_decisions, facts, procedures, episode_procedures, censors, working_memory
7. All indexes (after tables)
8. All triggers (after tables)

**Key deviations from research/008:**

1. **HNSW everywhere** (D1):
   ```sql
   -- CORRECT (use this)
   CREATE INDEX idx_decisions_embedding ON brain.decisions
       USING hnsw(embedding vector_cosine_ops);

   -- WRONG (do NOT copy from 008)
   -- CREATE INDEX ... USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);
   ```

2. **Guardrails unique constraint** (D5):
   ```sql
   -- In brain.guardrails:
   -- REMOVE: name VARCHAR(200) NOT NULL UNIQUE
   -- REPLACE WITH:
   name VARCHAR(200) NOT NULL,
   -- ... then in table constraints:
   UNIQUE(agent_id, name)
   ```

3. **Updated_at trigger** (D6):
   ```sql
   CREATE OR REPLACE FUNCTION update_timestamp()
   RETURNS trigger AS $$
   BEGIN
       NEW.updated_at = NOW();
       RETURN NEW;
   END;
   $$ LANGUAGE plpgsql;

   -- Apply to every table with updated_at:
   -- brain.decisions, heart.facts, heart.procedures, heart.censors,
   -- heart.working_memory, nous_system.agents
   CREATE TRIGGER set_updated_at BEFORE UPDATE ON brain.decisions
       FOR EACH ROW EXECUTE FUNCTION update_timestamp();
   ```

4. **CHECK constraints on enum-like columns** (D7):
   ```sql
   -- brain.decisions:
   stakes VARCHAR(20) NOT NULL CHECK (stakes IN ('low', 'medium', 'high', 'critical')),
   category VARCHAR(50) NOT NULL CHECK (category IN ('architecture', 'process', 'tooling', 'security', 'integration')),
   outcome VARCHAR(20) DEFAULT 'pending' CHECK (outcome IN ('pending', 'success', 'partial', 'failure')),

   -- brain.guardrails:
   severity VARCHAR(20) NOT NULL DEFAULT 'warn' CHECK (severity IN ('warn', 'block', 'absolute')),

   -- heart.censors:
   action VARCHAR(20) NOT NULL DEFAULT 'warn' CHECK (action IN ('warn', 'block', 'absolute')),

   -- heart.episodes:
   outcome VARCHAR(50) CHECK (outcome IN ('success', 'partial', 'failure', 'ongoing', 'abandoned')),

   -- heart.episode_procedures:
   effectiveness VARCHAR(20) CHECK (effectiveness IN ('helped', 'neutral', 'hindered')),
   ```

5. **Generated tsvector columns for FTS** (D8):
   ```sql
   -- brain.decisions — add column:
   search_tsv tsvector GENERATED ALWAYS AS (
       to_tsvector('english', COALESCE(description, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(pattern, ''))
   ) STORED,

   -- Then index on the column, not the expression:
   CREATE INDEX idx_decisions_fts ON brain.decisions USING GIN(search_tsv);

   -- Same pattern for: episodes, facts, procedures
   ```

6. **Additional indexes** (D14, D15):
   ```sql
   CREATE INDEX idx_decisions_agent_created ON brain.decisions(agent_id, created_at DESC);
   CREATE INDEX idx_frames_agent ON nous_system.frames(agent_id);
   ```

7. **DO NOT include**: `search_hybrid()` function (D4).

### `sql/seed.sql`

Use exactly as specified in 001-postgres-scaffold.md. The JSONB structures, frame definitions, and guardrail conditions are correct. Insertion order: agents first, then frames and guardrails.

### `pyproject.toml`

```toml
[project]
name = "nous"
version = "0.1.0"
description = "A cognitive agent framework built on Society of Mind principles"
requires-python = ">=3.12"
dependencies = [
    "sqlalchemy[asyncio]>=2.0,<3.0",
    "asyncpg>=0.30,<1.0",
    "pgvector>=0.3,<1.0",
    "pydantic>=2.0,<3.0",
    "pydantic-settings>=2.0,<3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

**Notes:** Upper-bounded dependency ranges. Minimal ruff config. pytest-asyncio auto mode.

### `nous/__init__.py`

```python
"""Nous — A cognitive agent framework built on Society of Mind principles."""

__version__ = "0.1.0"
```

### `nous/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOUS_", env_file=".env")

    db_url: str = "postgresql+asyncpg://nous:nous_dev_password@localhost:5432/nous"
    db_pool_size: int = 10
    db_max_overflow: int = 5
    agent_id: str = "nous-default"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    log_level: str = "info"
```

**Notes:** Uses `SettingsConfigDict` (idiomatic pydantic-settings v2). Adds `.env` file support.

### `nous/storage/database.py`

```python
"""Async database engine and session management."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nous.config import Settings


class Database:
    def __init__(self, settings: Settings) -> None:
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
            result = await conn.execute(
                text(
                    "SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name IN ('brain', 'heart', 'nous_system')"
                )
            )
            schemas = {row[0] for row in result}
            expected = {"brain", "heart", "nous_system"}
            if schemas != expected:
                missing = expected - schemas
                raise RuntimeError(f"Missing database schemas: {missing}")

    async def disconnect(self) -> None:
        """Dispose of connection pool."""
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an async session with automatic cleanup."""
        async with self.session_factory() as session:
            yield session

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()
```

**Key fixes from reviews:**
- `from sqlalchemy import text` (arch P1-1)
- `raise RuntimeError` instead of `assert` (D2)
- f-string bug fixed (arch P2-2)
- Session returns async context manager (arch P2-7)
- Database implements `__aenter__`/`__aexit__` (arch P2-7)

### `nous/storage/models.py`

All 18 ORM models. Key patterns to follow:

```python
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint, CheckConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Single declarative base for all schemas."""
    pass


# --- nous_system schema ---

class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = {"schema": "nous_system"}

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    active: Mapped[bool] = mapped_column(Boolean, server_default="true")
    last_active: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


# --- brain schema ---

class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = {"schema": "brain"}

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=func.gen_random_uuid())
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text)
    pattern: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    stakes: Mapped[str] = mapped_column(String(20), nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float)
    outcome: Mapped[str] = mapped_column(String(20), server_default="pending")
    outcome_result: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column()
    embedding = mapped_column(Vector(1536), nullable=True)
    # search_tsv is GENERATED — do not map, read-only
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Relationships
    tags: Mapped[list["DecisionTag"]] = relationship(back_populates="decision", cascade="all, delete-orphan")
    reasons: Mapped[list["DecisionReason"]] = relationship(back_populates="decision", cascade="all, delete-orphan")
    bridge: Mapped["DecisionBridge | None"] = relationship(back_populates="decision", cascade="all, delete-orphan", uselist=False)
    thoughts: Mapped[list["Thought"]] = relationship(back_populates="decision", cascade="all, delete-orphan")


# --- heart schema (cross-schema FK example) ---

class EpisodeDecision(Base):
    __tablename__ = "episode_decisions"
    __table_args__ = {"schema": "heart"}

    episode_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("heart.episodes.id", ondelete="CASCADE"), primary_key=True
    )
    decision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("brain.decisions.id", ondelete="CASCADE"), primary_key=True
    )

# ... remaining 14 models follow same patterns
```

**Implementation notes:**
- Single `Base` class shared across all schemas
- Schema specified via `__table_args__ = {"schema": "..."}`
- Cross-schema FKs use full qualified path: `ForeignKey("brain.decisions.id")`
- `Vector(1536)` column NOT wrapped in `Mapped[]` (pgvector quirk)
- Generated `tsvector` columns are NOT mapped — they're read-only DB-side
- Composite PKs use `primary_key=True` on multiple columns
- `agent_id` columns are plain VARCHAR, NOT FK to nous_system.agents (intentional — allows records before agent registration, avoids circular schema deps)

### `tests/conftest.py`

```python
"""Test fixtures using real Postgres via docker-compose."""

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from nous.config import Settings
from nous.storage.database import Database


@pytest_asyncio.fixture(scope="session")
async def db():
    """Session-scoped database connection pool."""
    settings = Settings()
    database = Database(settings)
    await database.connect()
    yield database
    await database.disconnect()


@pytest_asyncio.fixture
async def session(db):
    """Function-scoped session with SAVEPOINT isolation.

    Tests can call session.commit() freely — everything is rolled back
    after each test via the outer transaction.
    """
    async with db.engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)

        # Start a SAVEPOINT
        nested = await conn.begin_nested()

        @event.listens_for(session.sync_session, "after_transaction_end")
        def restart_savepoint(sess, transaction):
            nonlocal nested
            if transaction.nested and not transaction._parent.nested:
                nested = conn.sync_connection.begin_nested()

        yield session

        # Roll back the outer transaction — undoes everything
        await session.close()
        await trans.rollback()
```

**Key design:**
- `db` is session-scoped (one pool for all tests, D13)
- `session` is function-scoped with SAVEPOINT pattern (D3)
- Tests can freely `commit()` without leaking data

### `tests/test_database.py`

Tests to implement:
1. `test_connection` — can connect to Postgres
2. `test_schemas_exist` — brain, heart, nous_system schemas present
3. `test_extensions` — vector and pg_trgm extensions installed
4. `test_all_tables_exist` — all 18 tables exist with correct schemas
5. `test_seed_agent` — default agent exists with correct config
6. `test_seed_frames` — 6 cognitive frames exist
7. `test_seed_guardrails` — 4 guardrails exist with correct conditions
8. `test_updated_at_trigger` — updating a row auto-updates `updated_at`

### `tests/test_models.py`

Tests to implement:
1. `test_create_decision` — insert via ORM, read back, verify all fields
2. `test_create_episode` — insert episode with tags array, verify GIN-indexable
3. `test_create_fact` — insert fact with embedding vector, verify storage
4. `test_decision_with_tags` — insert decision + tags via relationship, query by tag
5. `test_decision_with_reasons` — insert with typed reasons, verify cascade
6. `test_guardrail_jsonb` — read guardrail, parse JSONB condition
7. `test_cross_schema_relationship` — episode_decisions linking heart.episode + brain.decision
8. `test_check_constraints` — verify invalid enum values are rejected
9. `test_null_embedding` — insert record with NULL embedding (no error)
10. `test_unique_guardrail_per_agent` — same name OK for different agents, duplicate fails for same agent

---

## Risk Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| ivfflat on empty tables returns 0 results | **Critical** | D1: HNSW everywhere |
| assert stripped by `-O` flag, schema check silently passes | **Critical** | D2: raise RuntimeError |
| Test data leaks between tests | **High** | D3: SAVEPOINT pattern |
| search_hybrid() broken (nonexistent column) | **High** | D4: Excluded from init.sql |
| Guardrail names collide across agents | **High** | D5: UNIQUE(agent_id, name) |
| updated_at timestamps permanently stale | **Medium** | D6: Trigger function |
| Invalid enum values silently accepted | **Medium** | D7: CHECK constraints |
| Docker init scripts ignored after first run | **Medium** | Documented in docker-compose.yml comment |
| Embedding dimension lock-in (vector(1536)) | **Low** | Accepted trade-off. Documented in init.sql comment. Changing models requires ALTER TABLE — acceptable at this scale. |
| Port 5432 conflicts with local Postgres | **Low** | `${DB_PORT:-5432}` is configurable via .env |

---

## Acceptance Criteria Checklist

- [ ] `docker compose up -d postgres` starts Postgres with pgvector and pg_trgm
- [ ] `sql/init.sql` creates 3 schemas, 18 tables, all indexes (HNSW, not ivfflat), trigger function
- [ ] `sql/seed.sql` populates 1 default agent, 6 frames, 4 guardrails
- [ ] `nous/config.py` loads settings from env vars with `NOUS_` prefix
- [ ] `nous/storage/database.py` provides async connection pool with proper error handling
- [ ] `nous/storage/models.py` has ORM models for all 18 tables
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Clean project structure matching the spec tree (minus `migrations.py`)
- [ ] No `assert` in production code
- [ ] All vector indexes use HNSW
- [ ] All enum-like columns have CHECK constraints
- [ ] All tables with `updated_at` have trigger attached

---

## What's NOT in This Phase

Per the spec's non-goals, confirmed by all reviewers:

- No REST API or MCP interface
- No Brain/Heart business logic
- No embedding generation (needs API key)
- No Dockerfile for Nous agent
- No `search_hybrid()` function
- No `migrations.py` / Alembic
- No storage abstraction layer or factory pattern (from research/004)
- No dashboard

---

*Three minds reviewed. One plan emerged. Build the foundation.*
