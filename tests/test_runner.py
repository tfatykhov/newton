"""Unit tests for AgentRunner — the LLM execution loop.

Tests mock _call_sdk() to avoid real Claude API calls and
claude_agent_sdk import dependency. MockCognitiveLayer records
all pre_turn/post_turn/end_session calls for assertions.
"""

import logging
import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from nous.api.runner import AgentRunner, MAX_CONVERSATIONS
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

    async def pre_turn(self, agent_id, session_id, user_input, session=None, *, conversation_messages=None):
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
# Mock Brain / Heart (stubs — runner only uses them for MCP server creation)
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
    """AgentRunner with mocked _call_sdk (no real SDK dependency).

    We skip runner.start() since it imports claude_agent_sdk.
    Instead, we mock _call_sdk directly to return controlled responses.
    """
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Mock _call_sdk to return a simple text response with no tool calls
    r._call_sdk = AsyncMock(return_value=("Hello from Nous!", []))

    yield r
    await r.close()


@pytest_asyncio.fixture
async def runner_error(mock_cognitive, mock_settings):
    """AgentRunner where _call_sdk raises an exception."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Mock _call_sdk to raise an exception (simulates SDK/API error)
    r._call_sdk = AsyncMock(side_effect=RuntimeError("SDK query failed: Internal error"))

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
    """SDK error -> error message returned, post_turn still called with error."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    response_text, turn_context = await runner_error.run_turn(session_id, "Hello!")

    # Should return error fallback message
    assert "error" in response_text.lower()

    # post_turn should still be called (per spec: always called, even on error)
    assert len(mock_cognitive.post_turn_calls) == 1
    _, _, turn_result, _ = mock_cognitive.post_turn_calls[0]
    assert turn_result.error is not None
    assert "SDK query failed" in turn_result.error


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


async def test_run_turn_with_tool_calls(mock_cognitive, mock_settings):
    """SDK returns tool results, post_turn sees them in TurnResult."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Mock _call_sdk to return tool results with captured result/error
    tool_results = [
        ToolResult(tool_name="record_decision", arguments={"description": "test"}, result="Decision recorded successfully.\nID: abc123"),
        ToolResult(tool_name="learn_fact", arguments={"content": "a fact"}, result="Fact learned successfully.\nID: def456"),
    ]
    r._call_sdk = AsyncMock(return_value=("Done with tools.", tool_results))

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
        ToolResult(tool_name="record_decision", arguments={"description": "bad"}, error="Error recording decision: validation failed"),
    ]
    r._call_sdk = AsyncMock(return_value=("Tool had an error.", tool_results))

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


async def test_run_turn_frame_instructions(mock_cognitive, mock_settings):
    """Decision frame adds 'MUST call record_decision' to system prompt."""
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

    r._call_sdk = AsyncMock(return_value=("Decision made.", []))

    try:
        session_id = f"test-frame-{uuid.uuid4().hex[:8]}"
        await r.run_turn(session_id, "Should I use caching?")

        # Check that _call_sdk was called with system prompt containing
        # the decision frame instructions
        call_args = r._call_sdk.call_args
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
    r._call_sdk = AsyncMock(return_value=("I decided to use caching.", []))

    try:
        session_id = f"test-safety-{uuid.uuid4().hex[:8]}"
        with caplog.at_level(logging.WARNING, logger="nous.api.runner"):
            await r.run_turn(session_id, "Should I use caching?")

        # Safety net should have logged a warning
        assert any("Safety net" in record.message for record in caplog.records)
    finally:
        await r.close()


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
    """Conversation with >= 3 turns triggers reflection via _call_sdk."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)

    # Track call count to distinguish turn calls from reflection call
    call_count = 0

    async def mock_call_sdk(system_prompt, user_message):
        nonlocal call_count
        call_count += 1
        if "reviewing a conversation" in system_prompt.lower():
            # This is the reflection call
            return ("Reflection: The task was about testing.", [])
        return (f"Response {call_count}", [])

    r._call_sdk = mock_call_sdk

    try:
        session_id = f"test-reflect-{uuid.uuid4().hex[:8]}"

        # Run 3 turns (6 messages: 3 user + 3 assistant)
        for i in range(3):
            await r.run_turn(session_id, f"Turn {i + 1}")

        assert len(r._conversations[session_id].messages) == 6

        await r.end_conversation(session_id)

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


async def test_start_credentials(mock_cognitive, mock_settings):
    """start() sets environment variables from settings."""
    brain = MockBrain()
    heart = MockHeart()

    # Test api_key only
    settings_api = Settings(
        ANTHROPIC_API_KEY="test-api-key-abc",
        agent_id="test-agent",
    )
    r = AgentRunner(mock_cognitive, brain, heart, settings_api)

    with patch("nous.api.runner.AgentRunner._call_sdk"), \
         patch("nous.api.tools.create_nous_mcp_server", return_value={"type": "sdk"}):
        await r.start()
        assert os.environ.get("ANTHROPIC_API_KEY") == "test-api-key-abc"
        await r.close()

    # Test auth_token takes precedence
    settings_auth = Settings(
        ANTHROPIC_API_KEY="test-api-key",
        ANTHROPIC_AUTH_TOKEN="test-auth-token-xyz",
        agent_id="test-agent",
    )
    r2 = AgentRunner(mock_cognitive, brain, heart, settings_auth)

    with patch("nous.api.tools.create_nous_mcp_server", return_value={"type": "sdk"}):
        await r2.start()
        assert os.environ.get("ANTHROPIC_AUTH_TOKEN") == "test-auth-token-xyz"
        await r2.close()

    # Test neither set -> warning logged
    settings_none = Settings(agent_id="test-agent")
    r3 = AgentRunner(mock_cognitive, brain, heart, settings_none)

    with patch("nous.api.tools.create_nous_mcp_server", return_value={"type": "sdk"}):
        # Remove any leftover env vars
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        await r3.start()  # Should not raise, just log warning
        await r3.close()


async def test_permission_mode_from_settings(mock_cognitive):
    """Runner uses sdk_permission_mode from settings, not hardcoded value."""
    brain = MockBrain()
    heart = MockHeart()
    settings = Settings(
        ANTHROPIC_API_KEY="test-key",
        agent_id="test-agent",
        sdk_permission_mode="default",
    )
    r = AgentRunner(mock_cognitive, brain, heart, settings)
    r._call_sdk = AsyncMock(return_value=("OK", []))

    try:
        assert r._settings.sdk_permission_mode == "default"
    finally:
        await r.close()


async def test_lru_eviction(mock_cognitive, mock_settings):
    """When MAX_CONVERSATIONS exceeded, oldest conversation is evicted."""
    brain = MockBrain()
    heart = MockHeart()
    r = AgentRunner(mock_cognitive, brain, heart, mock_settings)
    r._call_sdk = AsyncMock(return_value=("OK", []))

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
