"""Unit tests for AgentRunner -- the direct Anthropic API execution loop.

Tests mock _tool_loop() (for run_turn tests) or _call_api() (for
reflection/credential tests) to avoid real API calls.
MockCognitiveLayer records all pre_turn/post_turn/end_session calls.

Migrated from _call_sdk mocks to _tool_loop/_call_api mocks for
005.2 direct API rewrite.
"""

import logging
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from nous.api.runner import AgentRunner, ApiResponse, MAX_CONVERSATIONS
from nous.cognitive.schemas import FrameSelection, ToolResult, TurnContext, TurnResult
from nous.config import Settings


# ---------------------------------------------------------------------------
# Mock CognitiveLayer
# ---------------------------------------------------------------------------


class MockCognitiveLayer:
    """Returns preset TurnContext from pre_turn(), records all hook calls."""

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

    async def pre_turn(self, agent_id, session_id, user_input, session=None, *, conversation_messages=None, user_id=None, user_display_name=None):
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
# Mock Brain / Heart stubs
# ---------------------------------------------------------------------------


class MockBrain:
    """Stub Brain for runner constructor."""

    async def close(self):
        pass


class MockHeart:
    """Stub Heart for runner constructor."""

    async def close(self):
        pass


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
        ANTHROPIC_API_KEY="test-key-123",
        agent_id="test-agent",
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
    )


@pytest_asyncio.fixture
async def runner(mock_cognitive, mock_settings):
    """AgentRunner with mocked _tool_loop (no real API calls).

    _tool_loop returns the same (text, list[ToolResult]) shape that
    run_turn() expects. We skip start() since it creates a real httpx client.
    """
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Mock _tool_loop to return a simple text response with no tool calls
    r._tool_loop = AsyncMock(return_value=("Hello from Nous!", []))

    yield r
    await r.close()


@pytest_asyncio.fixture
async def runner_error(mock_cognitive, mock_settings):
    """AgentRunner where _tool_loop raises an exception."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Mock _tool_loop to raise an exception (simulates API error)
    r._tool_loop = AsyncMock(side_effect=RuntimeError("API call failed: Internal error"))

    yield r
    await r.close()


# ---------------------------------------------------------------------------
# Basic run_turn tests
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
    """API error -> error message returned, post_turn still called with error."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    response_text, turn_context = await runner_error.run_turn(session_id, "Hello!")

    # Should return error fallback message
    assert "error" in response_text.lower()

    # post_turn should still be called (per spec: always called, even on error)
    assert len(mock_cognitive.post_turn_calls) == 1
    _, _, turn_result, _ = mock_cognitive.post_turn_calls[0]
    assert turn_result.error is not None
    assert "API call failed" in turn_result.error


async def test_run_turn_history_capped(runner):
    """Messages capped at last 20 for API call via _format_messages."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"

    # Send 15 turns (30 messages total)
    for i in range(15):
        await runner.run_turn(session_id, f"Message {i}")

    conv = runner._conversations[session_id]
    formatted = runner._format_messages(conv)
    # Should be capped at 20 messages
    assert len(formatted) <= 20


# ---------------------------------------------------------------------------
# Tool call tests (mock _tool_loop returns ToolResults)
# ---------------------------------------------------------------------------


async def test_run_turn_with_tool_calls(mock_cognitive, mock_settings):
    """Tool loop returns tool results, post_turn sees them in TurnResult."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Mock _tool_loop to return tool results with captured result/error
    tool_results = [
        ToolResult(
            tool_name="record_decision",
            arguments={"description": "test"},
            result="Decision recorded successfully.\nID: abc123",
        ),
        ToolResult(
            tool_name="learn_fact",
            arguments={"content": "a fact"},
            result="Fact learned successfully.\nID: def456",
        ),
    ]
    r._tool_loop = AsyncMock(return_value=("Done with tools.", tool_results))

    try:
        session_id = f"test-tools-{uuid.uuid4().hex[:8]}"
        response_text, _ = await r.run_turn(session_id, "Do something with tools")

        assert response_text == "Done with tools."

        # post_turn should receive the tool results
        assert len(mock_cognitive.post_turn_calls) == 1
        _, _, turn_result, _ = mock_cognitive.post_turn_calls[0]
        assert len(turn_result.tool_results) == 2
        assert turn_result.tool_results[0].tool_name == "record_decision"
        assert turn_result.tool_results[0].result is not None
        assert "Decision recorded" in turn_result.tool_results[0].result
        assert turn_result.tool_results[1].tool_name == "learn_fact"
        assert turn_result.tool_results[1].result is not None
    finally:
        await r.close()


async def test_run_turn_with_tool_error(mock_cognitive, mock_settings):
    """Tool error captured in ToolResult.error field."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    tool_results = [
        ToolResult(
            tool_name="record_decision",
            arguments={"description": "bad"},
            error="Error recording decision: validation failed",
        ),
    ]
    r._tool_loop = AsyncMock(return_value=("Tool had an error.", tool_results))

    try:
        session_id = f"test-tool-err-{uuid.uuid4().hex[:8]}"
        response_text, _ = await r.run_turn(session_id, "Try something broken")

        assert len(mock_cognitive.post_turn_calls) == 1
        _, _, turn_result, _ = mock_cognitive.post_turn_calls[0]
        assert len(turn_result.tool_results) == 1
        assert turn_result.tool_results[0].error is not None
        assert "validation failed" in turn_result.tool_results[0].error
    finally:
        await r.close()


# ---------------------------------------------------------------------------
# Frame instruction tests
# ---------------------------------------------------------------------------


async def test_run_turn_frame_instructions(mock_cognitive, mock_settings):
    """Decision frame adds 'MUST call record_decision' to system prompt.

    Verifies _tool_loop receives the system prompt with frame instructions.
    """
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Set frame to "decision"
    mock_cognitive.preset_context = TurnContext(
        system_prompt="You are Nous.",
        frame=FrameSelection(
            frame_id="decision",
            frame_name="Decision",
            confidence=0.95,
            match_method="pattern",
        ),
        decision_id="test-decision-123",
        active_censors=[],
        context_token_estimate=100,
    )

    r._tool_loop = AsyncMock(return_value=("Decision made.", []))

    try:
        session_id = f"test-frame-{uuid.uuid4().hex[:8]}"
        await r.run_turn(session_id, "Should I use caching?")

        # Check that _tool_loop was called with system prompt containing
        # the decision frame instructions
        call_args = r._tool_loop.call_args
        system_prompt = call_args.kwargs.get("system_prompt") or call_args.args[0]
        assert "MUST call" in system_prompt
        assert "record_decision" in system_prompt
    finally:
        await r.close()


async def test_run_turn_safety_net(mock_cognitive, mock_settings, caplog):
    """Missed record_decision in decision frame -> warning logged."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Set frame to "decision" (in _DECISION_FRAMES)
    mock_cognitive.preset_context = TurnContext(
        system_prompt="You are Nous.",
        frame=FrameSelection(
            frame_id="decision",
            frame_name="Decision",
            confidence=0.95,
            match_method="pattern",
        ),
        decision_id="test-decision-456",
        active_censors=[],
        context_token_estimate=100,
    )

    # Return response with NO tool calls (record_decision not called)
    r._tool_loop = AsyncMock(return_value=("I decided to use caching.", []))

    try:
        session_id = f"test-safety-{uuid.uuid4().hex[:8]}"
        with caplog.at_level(logging.WARNING, logger="nous.api.runner"):
            await r.run_turn(session_id, "Should I use caching?")

        # Safety net should have logged a warning
        assert any("Safety net" in record.message for record in caplog.records)
    finally:
        await r.close()


# ---------------------------------------------------------------------------
# Conversation lifecycle tests
# ---------------------------------------------------------------------------


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


async def test_end_conversation_with_reflection(mock_cognitive, mock_settings):
    """Conversation with >= 3 turns triggers reflection via _call_api.

    run_turn calls _tool_loop, end_conversation calls _call_api directly.
    """
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Mock _tool_loop for the 3 run_turn calls
    turn_counter = 0

    async def mock_tool_loop(system_prompt, conversation, frame_id):
        nonlocal turn_counter
        turn_counter += 1
        return (f"Response {turn_counter}", [])

    r._tool_loop = mock_tool_loop

    # Mock _call_api for the reflection call in end_conversation
    r._call_api = AsyncMock(return_value=ApiResponse(
        content=[{"type": "text", "text": "Reflection: The task was about testing."}],
        stop_reason="end_turn",
    ))

    try:
        session_id = f"test-reflect-{uuid.uuid4().hex[:8]}"

        # Run 3 turns (6 messages: 3 user + 3 assistant)
        for i in range(3):
            await r.run_turn(session_id, f"Turn {i + 1}")

        assert len(r._conversations[session_id].messages) == 6

        await r.end_conversation(session_id)

        # _call_api should have been called for reflection
        r._call_api.assert_called_once()
        call_kwargs = r._call_api.call_args
        reflection_system = call_kwargs.kwargs.get("system_prompt") or call_kwargs.args[0]
        assert "reviewing a conversation" in reflection_system.lower()

        # end_session should have been called with reflection text
        assert len(mock_cognitive.end_session_calls) == 1
        _, _, reflection = mock_cognitive.end_session_calls[0]
        assert reflection is not None
        assert "Reflection" in reflection
    finally:
        await r.close()


async def test_end_conversation_nonexistent(runner, mock_cognitive):
    """Ending nonexistent conversation doesn't error."""
    session_id = f"nonexistent-{uuid.uuid4().hex[:8]}"
    # Should not raise
    await runner.end_conversation(session_id)
    # end_session should still be called
    assert len(mock_cognitive.end_session_calls) == 1


# ---------------------------------------------------------------------------
# Credential / start() tests
# ---------------------------------------------------------------------------


async def test_start_credentials_api_key(mock_cognitive):
    """start() creates httpx client with x-api-key header when API key set."""
    brain = MockBrain()
    heart = MockHeart()
    settings = Settings(
        ANTHROPIC_API_KEY="test-api-key-abc",
        agent_id="test-agent",
    )
    r = AgentRunner(mock_cognitive, brain, heart, settings)

    await r.start()
    try:
        # Verify httpx client was created with x-api-key header
        assert r._http is not None
        assert r._http.headers.get("x-api-key") == "test-api-key-abc"
        assert r._http.headers.get("anthropic-version") == "2023-06-01"
        assert r._http.headers.get("content-type") == "application/json"
        # No authorization header when only API key set
        assert "authorization" not in r._http.headers
    finally:
        await r.close()


async def test_start_credentials_auth_token(mock_cognitive):
    """start() uses Bearer token (takes precedence over API key)."""
    brain = MockBrain()
    heart = MockHeart()
    settings = Settings(
        ANTHROPIC_API_KEY="test-api-key",
        ANTHROPIC_AUTH_TOKEN="test-auth-token-xyz",
        agent_id="test-agent",
    )
    r = AgentRunner(mock_cognitive, brain, heart, settings)

    await r.start()
    try:
        # Bearer token takes precedence
        assert r._http is not None
        assert r._http.headers.get("authorization") == "Bearer test-auth-token-xyz"
        # x-api-key should NOT be set when auth_token is present
        assert "x-api-key" not in r._http.headers
    finally:
        await r.close()


async def test_start_credentials_none(mock_cognitive, caplog):
    """start() with no credentials logs a warning but doesn't raise."""
    brain = MockBrain()
    heart = MockHeart()
    settings = Settings(agent_id="test-agent")
    r = AgentRunner(mock_cognitive, brain, heart, settings)

    with caplog.at_level(logging.WARNING, logger="nous.api.runner"):
        await r.start()

    try:
        # Should have logged a warning about missing credentials
        assert any("API calls will fail" in record.message for record in caplog.records)
        # Client should still be created (just without auth headers)
        assert r._http is not None
        assert r._http.headers.get("anthropic-version") == "2023-06-01"
    finally:
        await r.close()


# ---------------------------------------------------------------------------
# LRU eviction test
# ---------------------------------------------------------------------------


async def test_lru_eviction(mock_cognitive, mock_settings):
    """When MAX_CONVERSATIONS exceeded, oldest conversation is evicted."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)
    r._tool_loop = AsyncMock(return_value=("OK", []))

    try:
        # Create MAX_CONVERSATIONS + 1 conversations
        session_ids = []
        for i in range(MAX_CONVERSATIONS + 1):
            sid = f"evict-{i:04d}"
            session_ids.append(sid)
            await r.run_turn(sid, f"Message {i}")

        # Should not exceed MAX_CONVERSATIONS
        assert len(r._conversations) <= MAX_CONVERSATIONS

        # The first session should have been evicted
        assert session_ids[0] not in r._conversations
        # The last session should still exist
        assert session_ids[-1] in r._conversations
    finally:
        await r.close()
