# Research Note 004: Storage Architecture

*One brain, swappable storage backends.*

## Decision

**PostgreSQL + pgvector as primary storage. Abstract storage layer for backend portability.**

Confidence: 0.85. Postgres is battle-tested, cloud-native, and unifies structured + vector data. The abstraction layer ensures we can swap to Qdrant, Weaviate, or anything else without touching cognitive code.

## Design Principles

1. **Single storage abstraction** — All memory types go through one API
2. **Backend-swappable** — Postgres default, but plug in anything
3. **Cloud-ready** — Works with managed services (RDS, Neon, Supabase) or local Docker
4. **Cognition Engines compatible** — CE remains the decision subsystem, Newton wraps it

## Storage Abstraction

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class MemoryRecord:
    """Universal memory record across all types."""
    id: str
    type: str              # "decision", "episode", "fact", "kline", "working"
    content: str           # The actual content
    embedding: list[float] | None = None
    metadata: dict[str, Any] = None
    created_at: str = None
    updated_at: str = None
    agent_id: str = None
    tags: list[str] = None
    
@dataclass  
class SearchResult:
    record: MemoryRecord
    score: float
    source: str            # Which backend returned this

class MemoryStore(ABC):
    """Abstract storage backend for Newton's memory."""
    
    @abstractmethod
    async def store(self, record: MemoryRecord) -> str:
        """Store a memory record. Returns ID."""
        ...
    
    @abstractmethod
    async def get(self, id: str) -> MemoryRecord | None:
        """Retrieve by ID."""
        ...
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        embedding: list[float] | None = None,
        type_filter: str | None = None,
        tag_filter: list[str] | None = None,
        limit: int = 10,
        mode: str = "hybrid"  # "semantic", "keyword", "hybrid"
    ) -> list[SearchResult]:
        """Search memory. Supports semantic, keyword, or hybrid."""
        ...
    
    @abstractmethod
    async def update(self, id: str, updates: dict[str, Any]) -> bool:
        """Partial update."""
        ...
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a record."""
        ...
    
    @abstractmethod
    async def list(
        self,
        type_filter: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[MemoryRecord]:
        """List records with filters."""
        ...


class EmbeddingProvider(ABC):
    """Abstract embedding generation."""
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...
```

## Backend Implementations

### PostgresMemoryStore (Primary)

```python
class PostgresMemoryStore(MemoryStore):
    """PostgreSQL + pgvector backend."""
    
    # Single table with JSONB metadata + vector column
    # GIN index on tags, GiST index on embedding
    # Full-text search via tsvector for keyword mode
    # Hybrid = RRF(semantic_rank, keyword_rank)
```

**Schema:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),      -- or 768, configurable
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    agent_id VARCHAR(100),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_memories_type ON memories(type);
CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_memories_tsv ON memories USING GIN(content_tsv);
```

**Hybrid search in one query:**
```sql
WITH semantic AS (
    SELECT id, 1 - (embedding <=> $1) AS score
    FROM memories
    WHERE type = $2
    ORDER BY embedding <=> $1
    LIMIT 20
),
keyword AS (
    SELECT id, ts_rank(content_tsv, plainto_tsquery($3)) AS score
    FROM memories
    WHERE content_tsv @@ plainto_tsquery($3) AND type = $2
    LIMIT 20
)
SELECT m.*, 
    COALESCE(s.score, 0) * 0.7 + COALESCE(k.score, 0) * 0.3 AS combined_score
FROM memories m
LEFT JOIN semantic s ON m.id = s.id
LEFT JOIN keyword k ON m.id = k.id
WHERE s.id IS NOT NULL OR k.id IS NOT NULL
ORDER BY combined_score DESC
LIMIT $4;
```

### SQLiteMemoryStore (Lightweight / Dev)

```python
class SQLiteMemoryStore(MemoryStore):
    """SQLite backend for local dev and testing.
    
    Uses sqlite-vec or manual cosine similarity.
    Good for: local dev, single-user, small datasets.
    """
```

### ChromaMemoryStore (Legacy Compat)

```python
class ChromaMemoryStore(MemoryStore):
    """ChromaDB backend for backward compatibility.
    
    Wraps existing ChromaDB collections.
    Migration path from CE's current setup.
    """
```

### QdrantMemoryStore (High-Performance Option)

```python
class QdrantMemoryStore(MemoryStore):
    """Qdrant backend for high-performance vector search.
    
    Best for: large datasets, low-latency requirements.
    Separate service but excellent performance.
    """
```

## Factory Pattern

```python
class MemoryStoreFactory:
    """Create the right backend from config."""
    
    _backends = {
        "postgres": PostgresMemoryStore,
        "sqlite": SQLiteMemoryStore,
        "chroma": ChromaMemoryStore,
        "qdrant": QdrantMemoryStore,
    }
    
    @classmethod
    def create(cls, config: dict) -> MemoryStore:
        backend = config.get("NEWTON_STORAGE", "postgres")
        store_class = cls._backends.get(backend)
        if not store_class:
            raise ValueError(f"Unknown storage backend: {backend}")
        return store_class(config)
```

**Environment config:**
```bash
# PostgreSQL (default)
NEWTON_STORAGE=postgres
NEWTON_DB_URL=postgresql://newton:secret@postgres:5432/newton

# SQLite (dev)
NEWTON_STORAGE=sqlite
NEWTON_DB_PATH=data/newton.db

# Qdrant (high-performance)
NEWTON_STORAGE=qdrant
NEWTON_QDRANT_URL=http://qdrant:6333
```

## Memory Types

| Type | What It Stores | Example |
|------|---------------|---------|
| `decision` | Choices made (via CE) | "Chose Postgres over ChromaDB" |
| `episode` | Multi-step experiences | "Debugging the SQLite migration" |
| `fact` | Learned information | "Tim prefers Celsius" |
| `kline` | Context bundles | K-line for trader analysis |
| `working` | Current session state | Active tasks, open threads |
| `procedure` | How-to knowledge | "Steps to deploy to production" |
| `censor` | Things NOT to do | "Don't push directly to main" |
| `frame` | Cognitive templates | Devil's Advocate frame |

## Cognition Engines Integration

Newton doesn't replace CE — it wraps it:

```mermaid
graph TB
    subgraph "Newton Memory Layer"
        MS[MemoryStore<br/>Unified API]
        MS --> PG[(Postgres)]
    end
    
    subgraph "Cognition Engines"
        CE[CSTP Server<br/>Decision Intelligence]
        CE --> PG
    end
    
    subgraph "Agent"
        A[Newton Agent]
        A -->|decisions| CE
        A -->|everything else| MS
    end
```

**Option A:** CE keeps its own storage, Newton stores non-decision memories separately. Simple but two systems.

**Option B:** CE migrates to Newton's MemoryStore as its backend. Unified but requires CE changes.

**Option C:** Both use same Postgres instance, different schemas. Shared infrastructure, independent logic.

**Recommendation: Option C** — Same database, `ce` schema for decisions, `newton` schema for everything else. One backup, one connection pool, independent evolution.

## Docker Compose

```yaml
version: "3.8"

services:
  newton:
    build: .
    environment:
      - NEWTON_STORAGE=postgres
      - NEWTON_DB_URL=postgresql://newton:${DB_PASSWORD}@postgres:5432/newton
      - CSTP_URL=http://cstp:9991
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy

  cstp:
    image: ghcr.io/tfatykhov/cognition-agent-decisions:latest
    environment:
      - CSTP_STORAGE=postgres
      - CSTP_DB_URL=postgresql://newton:${DB_PASSWORD}@postgres:5432/newton
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: newton
      POSTGRES_USER: newton
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U newton"]
      interval: 5s
      timeout: 5s
      retries: 5

  dashboard:
    image: ghcr.io/tfatykhov/cognition-dashboard:latest
    ports:
      - "8080:8080"
    environment:
      - CSTP_URL=http://cstp:9991

volumes:
  pgdata:
```

## Migration Path

1. **Phase 1:** Newton uses Postgres, CE stays on SQLite → separate stores
2. **Phase 2:** Add Postgres backend to CE (DecisionStore ABC makes this easy)
3. **Phase 3:** Both on same Postgres, shared infra, independent schemas
4. **Phase 4:** Optionally merge into unified MemoryStore

## Cloud Deployment Options

| Platform | Postgres | Newton Container | Cost |
|----------|----------|-----------------|------|
| Railway | Built-in Postgres | Docker deploy | ~$5-20/mo |
| Fly.io | Fly Postgres | Fly Machine | ~$5-15/mo |
| AWS | RDS/Aurora | ECS Fargate | ~$15-50/mo |
| Supabase | Built-in + pgvector | Any container host | Free tier available |
| Neon | Serverless Postgres | Any container host | Free tier, pay-per-query |
| Self-hosted | Docker Compose | Docker Compose | Just compute |

**Neon** is interesting for dev/demo — serverless Postgres with pgvector, generous free tier, scales to zero.

---

*Lesson from CE: the DecisionStore ABC saved us when migrating YAML → SQLite. Same pattern here — build the abstraction first, swap backends painlessly later.*
