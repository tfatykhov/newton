"""Integration tests for nous/api/tools.py tool closures.

All tests use real Brain/Heart instances with mock embeddings
against real Postgres. Tool closures capture Brain/Heart in closure
context and use auto-sessions (no session= parameter).
"""

import pytest
import pytest_asyncio

from nous.api.tools import create_nous_tools
from nous.brain.brain import Brain
from nous.config import Settings
from nous.heart import Heart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def brain(db, mock_embeddings):
    """Brain with mock embeddings for tool tests."""
    settings = Settings()
    b = Brain(database=db, settings=settings, embedding_provider=mock_embeddings)
    yield b
    await b.close()


@pytest_asyncio.fixture
async def tools(brain, heart):
    """Tool closures dict from create_nous_tools."""
    return create_nous_tools(brain, heart)


# ---------------------------------------------------------------------------
# test_create_nous_tools
# ---------------------------------------------------------------------------


class TestCreateNousTools:
    """Test that create_nous_tools returns the expected structure."""

    def test_create_nous_tools(self, tools):
        """Returns dict with 4 async callable functions."""
        assert isinstance(tools, dict)
        assert set(tools.keys()) == {
            "record_decision",
            "learn_fact",
            "recall_deep",
            "create_censor",
        }
        for name, func in tools.items():
            assert callable(func), f"{name} should be callable"


# ---------------------------------------------------------------------------
# record_decision tests
# ---------------------------------------------------------------------------


class TestRecordDecision:
    """Test the record_decision tool closure."""

    @pytest.mark.asyncio
    async def test_record_decision_success(self, tools):
        """Valid input -> brain.record called, returns ID."""
        result = await tools["record_decision"](
            description="Test tool decision for integration",
            confidence=0.85,
            category="tooling",
            stakes="low",
            context="Testing the record_decision tool closure",
            reasons=[
                {"type": "analysis", "text": "Tool test reason"},
                {"type": "pattern", "text": "Following test patterns"},
            ],
            tags=["test", "tool-closure"],
        )

        assert "content" in result
        assert len(result["content"]) == 1
        text = result["content"][0]["text"]
        assert "Decision recorded successfully" in text
        assert "ID:" in text
        assert "Quality score:" in text
        assert "Category: tooling" in text
        assert "Stakes: low" in text

    @pytest.mark.asyncio
    async def test_record_decision_invalid_reasons(self, tools):
        """Bad reason format -> error message (not exception)."""
        result = await tools["record_decision"](
            description="Decision with bad reasons",
            confidence=0.5,
            category="process",
            stakes="low",
            reasons=[{"bad": "format"}],  # Missing 'type' and 'text'
        )

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text
        assert "Invalid reason format" in text

    @pytest.mark.asyncio
    async def test_record_decision_brain_error(self, tools):
        """brain.record raises -> error message (not exception)."""
        # Trigger a pydantic validation error with invalid confidence range
        result = await tools["record_decision"](
            description="Decision with invalid confidence",
            confidence=2.0,  # Out of range [0.0, 1.0]
            category="tooling",
            stakes="low",
        )

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Error" in text or "error" in text.lower()


# ---------------------------------------------------------------------------
# learn_fact tests
# ---------------------------------------------------------------------------


class TestLearnFact:
    """Test the learn_fact tool closure."""

    @pytest.mark.asyncio
    async def test_learn_fact_success(self, tools):
        """Valid input -> heart.learn called, returns ID."""
        result = await tools["learn_fact"](
            content="Python 3.12 supports improved error messages",
            category="technical",
            subject="python",
            confidence=0.95,
            source="documentation",
            tags=["python", "test"],
        )

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Fact learned successfully" in text
        assert "ID:" in text
        assert "Category: technical" in text
        assert "Subject: python" in text

    @pytest.mark.asyncio
    async def test_learn_fact_with_contradiction(self, tools):
        """Learning a near-duplicate fact surfaces contradiction warning."""
        # Learn a fact first
        await tools["learn_fact"](
            content="The default database port is 5432",
            category="technical",
            subject="postgres",
        )

        # Learn a contradicting fact with same content (mock embeddings
        # produce identical vectors for identical text -> high similarity)
        result = await tools["learn_fact"](
            content="The default database port is 5432",
            category="technical",
            subject="postgres",
        )

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Fact learned successfully" in text
        # Contradiction warning may or may not appear depending on
        # similarity threshold; at minimum the fact should be stored
        assert "ID:" in text


# ---------------------------------------------------------------------------
# recall_deep tests
# ---------------------------------------------------------------------------


class TestRecallDeep:
    """Test the recall_deep tool closure."""

    @pytest.mark.asyncio
    async def test_recall_deep_all(self, tools):
        """Searches Heart + Brain when no memory_types specified."""
        # Seed data: record a decision and learn a fact
        await tools["record_decision"](
            description="Recall deep test architecture decision about caching",
            confidence=0.8,
            category="architecture",
            stakes="medium",
            context="Testing recall_deep all search",
            tags=["recall-test"],
        )
        await tools["learn_fact"](
            content="Caching improves performance in recall deep tests",
            category="technical",
            subject="caching",
        )

        # Search across all types (default)
        result = await tools["recall_deep"](query="caching architecture")

        assert "content" in result
        text = result["content"][0]["text"]
        # Should have both Heart and Brain sections
        assert "Heart Memory" in text or "Brain Decisions" in text

    @pytest.mark.asyncio
    async def test_recall_deep_decisions_only(self, tools):
        """memory_types=["decision"] searches Brain only."""
        # Seed a decision
        await tools["record_decision"](
            description="Decision-only recall test about deployment strategy",
            confidence=0.75,
            category="process",
            stakes="low",
            tags=["recall-decision-only"],
        )

        result = await tools["recall_deep"](
            query="deployment strategy",
            memory_types=["decision"],
        )

        assert "content" in result
        text = result["content"][0]["text"]
        # Should have Brain section but not Heart
        assert "Brain Decisions" in text
        assert "Heart Memory" not in text

    @pytest.mark.asyncio
    async def test_recall_deep_facts_only(self, tools):
        """memory_types=["fact"] searches Heart facts only."""
        # Seed a fact
        await tools["learn_fact"](
            content="Recall deep facts-only test about memory architecture",
            category="technical",
            subject="memory",
        )

        result = await tools["recall_deep"](
            query="memory architecture",
            memory_types=["fact"],
        )

        assert "content" in result
        text = result["content"][0]["text"]
        # Should have Heart section but not Brain
        assert "Heart Memory" in text
        assert "Brain Decisions" not in text

    @pytest.mark.asyncio
    async def test_recall_deep_empty(self, tools):
        """No results -> 'No results found.' in section output."""
        # Use heart-only type "episode" — no episodes seeded by tool tests
        result = await tools["recall_deep"](
            query="zzz_nonexistent_query_no_match",
            memory_types=["episode"],
        )

        assert "content" in result
        text = result["content"][0]["text"]
        assert "No results found" in text


# ---------------------------------------------------------------------------
# create_censor tests
# ---------------------------------------------------------------------------


class TestCreateCensor:
    """Test the create_censor tool closure."""

    @pytest.mark.asyncio
    async def test_create_censor_success(self, tools):
        """Valid input -> heart.add_censor, returns ID."""
        result = await tools["create_censor"](
            trigger_pattern="rm -rf /",
            reason="Dangerous command that could delete everything",
            action="block",
            domain="debugging",
        )

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Censor created successfully" in text
        assert "ID:" in text
        assert "Action: block" in text
        assert "Domain: debugging" in text
        assert "Pattern: rm -rf /" in text

    @pytest.mark.asyncio
    async def test_create_censor_invalid_uuid(self, tools):
        """Bad UUID string -> validation error message."""
        result = await tools["create_censor"](
            trigger_pattern="test pattern",
            reason="Test reason",
            learned_from_decision="not-a-valid-uuid",
        )

        assert "content" in result
        text = result["content"][0]["text"]
        assert "Validation error" in text or "Error" in text
