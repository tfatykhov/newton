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
