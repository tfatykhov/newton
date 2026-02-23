"""Test fixtures using real Postgres via docker-compose."""

import hashlib
import random

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from nous.config import Settings
from nous.storage.database import Database
from nous.storage.models import Guardrail

# ---------------------------------------------------------------------------
# Mock embedding provider (P1-4 fix: PRNG-seeded, L2-normalized vectors)
# ---------------------------------------------------------------------------


class MockEmbeddingProvider:
    """Returns deterministic, L2-normalized embeddings seeded from text hash.

    Uses PRNG seeded from SHA-256 hash of the input text to produce
    genuinely different 1536-dim vectors for different inputs. Unlike
    the naive hash-cycling approach, cosine similarity between unrelated
    texts is near zero while identical texts produce identical vectors.
    """

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        rng = random.Random(h)
        vec = [rng.gauss(0, 1) for _ in range(1536)]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    async def embed_near(self, text: str, noise: float = 0.05) -> list[float]:
        """Generate embedding similar to embed(text) but with controlled noise.

        Produces vectors with cosine similarity ~(1 - noise) to the base embedding.
        Used for testing near-duplicate detection and similarity thresholds.
        """
        base = await self.embed(text)
        rng = random.Random(f"{text}_near_{noise}")
        noisy = [v + rng.gauss(0, noise) for v in base]
        norm = sum(x * x for x in noisy) ** 0.5
        return [x / norm for x in noisy]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def db():
    """Session-scoped database connection pool."""
    settings = Settings()
    database = Database(settings)
    await database.connect()
    yield database
    await database.disconnect()


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Session-scoped settings instance."""
    return Settings()


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


# ---------------------------------------------------------------------------
# Brain fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embeddings() -> MockEmbeddingProvider:
    """Mock embedding provider for tests needing deterministic vectors."""
    return MockEmbeddingProvider()


@pytest_asyncio.fixture
async def heart(db, mock_embeddings):
    """Heart instance with mock embeddings for testing."""
    from nous.config import Settings
    from nous.heart import Heart

    settings = Settings()
    h = Heart(db, settings, embedding_provider=mock_embeddings)
    yield h
    await h.close()


GUARDRAIL_TEST_AGENT = "test-guardrail-agent"


@pytest_asyncio.fixture
async def seed_guardrails(session):
    """Insert the 4 default guardrails into the test session.

    Uses a test-specific agent_id to avoid unique constraint collisions
    with seed.sql data that is already loaded in the real database.

    Uses legacy JSONB format to test backward compatibility.
    """
    guardrails = [
        Guardrail(
            agent_id=GUARDRAIL_TEST_AGENT,
            name="no-high-stakes-low-confidence",
            description="Block high-stakes decisions with low confidence",
            condition={"stakes": "high", "confidence_lt": 0.5},  # Legacy JSONB
            severity="block",
            priority=100,
        ),
        Guardrail(
            agent_id=GUARDRAIL_TEST_AGENT,
            name="no-critical-without-review",
            description="Block critical-stakes without explicit review",
            condition={"stakes": "critical"},  # Legacy JSONB
            severity="block",
            priority=90,
        ),
        Guardrail(
            agent_id=GUARDRAIL_TEST_AGENT,
            name="require-reasons",
            description="Block decisions without at least one reason",
            condition={"reason_count_lt": 1},  # Legacy JSONB
            severity="block",
            priority=110,
        ),
        Guardrail(
            agent_id=GUARDRAIL_TEST_AGENT,
            name="low-quality-recording",
            description="Block low-quality decisions (missing tags/pattern)",
            condition={"quality_lt": 0.5},  # Legacy JSONB
            severity="block",
            priority=120,
        ),
    ]
    for g in guardrails:
        session.add(g)
    await session.flush()
    return guardrails
