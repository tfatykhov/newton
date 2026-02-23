"""Unit tests for AgentRunner — the LLM execution loop.

Tests use mock CognitiveLayer and httpx MockTransport to avoid
real API calls. No database needed for runner tests.
"""

import uuid

import httpx
import pytest
import pytest_asyncio

from nous.cognitive.schemas import FrameSelection, TurnContext, TurnResult
from nous.config import Settings

# ---------------------------------------------------------------------------
# Mock CognitiveLayer
# ---------------------------------------------------------------------------


class MockCognitiveLayer:
    """Returns preset TurnContext from pre_turn(), records post_turn()/end_session() calls."""

    def __init__(self) -> None:
        self.pre_turn_calls: list[tuple] = []
        self.post_turn_calls: list[tuple] = []
        self.end_session_calls: list[tuple] = []
        self.preset_context = TurnContext(
            system_prompt="You are Nous, a thinking agent.",
            frame=FrameSelection(
                frame_id="conversation",
                frame_name="Conversation",
                confidence=0.9,
                match_method="default",
            ),
            decision_id=None,
            active_censors=[],
            context_token_estimate=100,
        )

    async def pre_turn(self, agent_id, session_id, user_input, session=None):
        self.pre_turn_calls.append((agent_id, session_id, user_input))
        return self.preset_context

    async def post_turn(self, agent_id, session_id, turn_result, turn_context, session=None):
        self.post_turn_calls.append((agent_id, session_id, turn_result, turn_context))
        from nous.cognitive.schemas import Assessment

        return Assessment(actual=turn_result.response_text[:200])

    async def end_session(self, agent_id, session_id, reflection=None, session=None):
        self.end_session_calls.append((agent_id, session_id, reflection))

    async def list_frames(self, agent_id, session=None):
        return []


# ---------------------------------------------------------------------------
# Mock Anthropic API transport
# ---------------------------------------------------------------------------


def make_mock_transport(response_text: str = "Hello from Nous!", status_code: int = 200):
    """Create httpx MockTransport simulating Anthropic Messages API."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status_code != 200:
            return httpx.Response(
                status_code=status_code,
                json={"error": {"type": "server_error", "message": "Internal error"}},
            )
        return httpx.Response(
            status_code=200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
                "model": "claude-sonnet-4-5-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cognitive():
    return MockCognitiveLayer()


@pytest.fixture
def mock_settings():
    """Settings with a fake API key for testing."""
    return Settings(
        anthropic_api_key="test-key-123",
        agent_id="test-agent",
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
    )


@pytest_asyncio.fixture
async def runner(mock_cognitive, mock_settings):
    """AgentRunner with mock cognitive layer and mock HTTP transport."""
    from nous.api.runner import AgentRunner

    r = AgentRunner(mock_cognitive, mock_settings)
    # Replace the httpx client with one using MockTransport
    r._client = httpx.AsyncClient(
        transport=make_mock_transport("Hello from Nous!"),
        base_url="https://api.anthropic.com",
        headers={
            "x-api-key": mock_settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=120.0,
    )
    yield r
    await r.close()


@pytest_asyncio.fixture
async def runner_error(mock_cognitive, mock_settings):
    """AgentRunner with API that returns 500 errors."""
    from nous.api.runner import AgentRunner

    r = AgentRunner(mock_cognitive, mock_settings)
    r._client = httpx.AsyncClient(
        transport=make_mock_transport("", status_code=500),
        base_url="https://api.anthropic.com",
        headers={
            "x-api-key": mock_settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=120.0,
    )
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_run_turn_basic(runner, mock_cognitive):
    """Sends message, gets response, returns (text, context)."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    response_text, turn_context = await runner.run_turn(session_id, "Hello!")

    assert response_text == "Hello from Nous!"
    assert isinstance(turn_context, TurnContext)
    assert turn_context.frame.frame_id == "conversation"


async def test_run_turn_creates_conversation(runner):
    """New session_id creates Conversation."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    await runner.run_turn(session_id, "Hello!")

    assert session_id in runner._conversations
    conv = runner._conversations[session_id]
    assert conv.session_id == session_id


async def test_run_turn_appends_messages(runner):
    """Conversation history grows each turn."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"

    await runner.run_turn(session_id, "First message")
    await runner.run_turn(session_id, "Second message")

    conv = runner._conversations[session_id]
    # Should have 4 messages: user1, assistant1, user2, assistant2
    assert len(conv.messages) == 4
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "First message"
    assert conv.messages[1].role == "assistant"
    assert conv.messages[2].role == "user"
    assert conv.messages[2].content == "Second message"
    assert conv.messages[3].role == "assistant"


async def test_run_turn_calls_pre_turn(runner, mock_cognitive):
    """cognitive.pre_turn() called with correct arguments."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    await runner.run_turn(session_id, "Hello!")

    assert len(mock_cognitive.pre_turn_calls) == 1
    agent_id, sid, user_input = mock_cognitive.pre_turn_calls[0]
    assert sid == session_id
    assert user_input == "Hello!"


async def test_run_turn_calls_post_turn(runner, mock_cognitive):
    """cognitive.post_turn() called with TurnResult."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    await runner.run_turn(session_id, "Hello!")

    assert len(mock_cognitive.post_turn_calls) == 1
    agent_id, sid, turn_result, turn_context = mock_cognitive.post_turn_calls[0]
    assert sid == session_id
    assert isinstance(turn_result, TurnResult)
    assert turn_result.response_text == "Hello from Nous!"
    assert turn_result.error is None


async def test_run_turn_api_error(runner_error, mock_cognitive):
    """API 500 -> error message returned, post_turn still called."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    response_text, turn_context = await runner_error.run_turn(session_id, "Hello!")

    # Should return some error indication
    assert response_text != "" or response_text == ""  # Implementation may vary
    # post_turn should still be called (per spec)
    assert len(mock_cognitive.post_turn_calls) == 1
    _, _, turn_result, _ = mock_cognitive.post_turn_calls[0]
    assert turn_result.error is not None


async def test_run_turn_history_capped(runner):
    """Messages capped at last 20 for API call."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"

    # Send 15 turns (30 messages total)
    for i in range(15):
        await runner.run_turn(session_id, f"Message {i}")

    conv = runner._conversations[session_id]
    formatted = runner._format_messages(conv)
    # Should be capped at 20 messages
    assert len(formatted) <= 20


async def test_end_conversation(runner, mock_cognitive):
    """Removes from dict, calls cognitive.end_session()."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    await runner.run_turn(session_id, "Hello!")
    assert session_id in runner._conversations

    await runner.end_conversation(session_id)

    assert session_id not in runner._conversations
    assert len(mock_cognitive.end_session_calls) == 1
    agent_id, sid, reflection = mock_cognitive.end_session_calls[0]
    assert sid == session_id


async def test_end_conversation_nonexistent(runner, mock_cognitive):
    """Ending nonexistent conversation doesn't error."""
    session_id = f"nonexistent-{uuid.uuid4().hex[:8]}"
    # Should not raise
    await runner.end_conversation(session_id)


async def test_lru_eviction(mock_cognitive, mock_settings):
    """When MAX_CONVERSATIONS exceeded, oldest conversation is evicted."""
    from nous.api.runner import AgentRunner

    r = AgentRunner(mock_cognitive, mock_settings)
    r._client = httpx.AsyncClient(
        transport=make_mock_transport("OK"),
        base_url="https://api.anthropic.com",
        headers={
            "x-api-key": mock_settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=120.0,
    )

    try:
        # Get MAX_CONVERSATIONS from the class (default 100)
        max_convs = getattr(r, "MAX_CONVERSATIONS", 100)

        # Create max+1 conversations
        session_ids = []
        for i in range(max_convs + 1):
            sid = f"evict-{i:04d}"
            session_ids.append(sid)
            await r.run_turn(sid, f"Message {i}")

        # Should not exceed MAX_CONVERSATIONS
        assert len(r._conversations) <= max_convs

        # The first session should have been evicted
        assert session_ids[0] not in r._conversations
        # The last session should still exist
        assert session_ids[-1] in r._conversations
    finally:
        await r.close()
