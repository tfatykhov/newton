"""Agent runner — executes a single conversational turn.

Wires CognitiveLayer.pre_turn() and post_turn() around
the actual LLM call. This is the core execution loop.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx

from nous.cognitive.layer import CognitiveLayer
from nous.cognitive.schemas import TurnContext, TurnResult
from nous.config import Settings

logger = logging.getLogger(__name__)

MAX_CONVERSATIONS = 100
MAX_HISTORY_MESSAGES = 20


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """Tracks a multi-turn conversation."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    turn_contexts: list[TurnContext] = field(default_factory=list)


class AgentRunner:
    """Runs conversational turns with cognitive layer hooks.

    For v0.1.0, uses the Anthropic Messages API directly via httpx.
    Claude Agent SDK integration is a future enhancement.
    """

    REFLECTION_PROMPT = (
        "Review this conversation. Summarize briefly:\n"
        "1. What was the main task?\n"
        "2. What went well?\n"
        "3. What should be done differently?\n"
        '4. List any new facts learned as "learned: <fact>" lines (one per line).'
    )

    def __init__(
        self,
        cognitive: CognitiveLayer,
        settings: Settings,
    ) -> None:
        self._cognitive = cognitive
        self._settings = settings
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize the HTTP client for Anthropic API."""
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": self._settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def run_turn(
        self,
        session_id: str,
        user_message: str,
        agent_id: str | None = None,
    ) -> tuple[str, TurnContext]:
        """Execute a single conversational turn.

        Steps:
        1. Get or create Conversation for session_id
        2. Call cognitive.pre_turn() -> TurnContext
        3. Append user message to conversation history
        4. Call Anthropic Messages API
        5. Extract assistant response text
        6. Append assistant message to conversation history
        7. Call cognitive.post_turn() with TurnResult
        8. Return (response_text, turn_context)

        On API error: log, create TurnResult with error, still call post_turn.
        """
        _agent_id = agent_id or self._settings.agent_id
        conversation = self._get_or_create_conversation(session_id)

        # 2. Pre-turn
        turn_context = await self._cognitive.pre_turn(_agent_id, session_id, user_message)

        # 3. Append user message
        conversation.messages.append(Message(role="user", content=user_message))

        # 4-6. Call LLM
        response_text = ""
        error = None
        try:
            response_text = await self._call_llm(
                system_prompt=turn_context.system_prompt,
                messages=self._format_messages(conversation),
            )
            conversation.messages.append(Message(role="assistant", content=response_text))
        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            error = str(e)
            response_text = "I encountered an error processing your request. Please try again."
            conversation.messages.append(Message(role="assistant", content=response_text))

        # 7. Post-turn (always called, even on error)
        turn_result = TurnResult(response_text=response_text, error=error)
        await self._cognitive.post_turn(_agent_id, session_id, turn_result, turn_context)

        # Store context
        conversation.turn_contexts.append(turn_context)

        return response_text, turn_context

    async def end_conversation(self, session_id: str, agent_id: str | None = None) -> None:
        """End a conversation with reflection.

        1. If conversation has >= 3 turns, generate reflection via LLM
        2. Call cognitive.end_session(agent_id, session_id, reflection=...)
        3. Remove from self._conversations
        """
        _agent_id = agent_id or self._settings.agent_id
        conversation = self._conversations.get(session_id)

        reflection: str | None = None
        if conversation and len(conversation.messages) >= 6:  # 3 user + 3 assistant = 6 messages
            try:
                reflection = await self._call_llm(
                    system_prompt="You are reviewing a conversation to extract lessons learned.",
                    messages=self._format_messages(conversation)
                    + [{"role": "user", "content": self.REFLECTION_PROMPT}],
                )
            except Exception as e:
                logger.warning("Failed to generate reflection: %s", e)

        await self._cognitive.end_session(_agent_id, session_id, reflection=reflection)

        # Remove conversation
        self._conversations.pop(session_id, None)

    async def _call_llm(self, system_prompt: str, messages: list[dict]) -> str:
        """Call Anthropic Messages API."""
        if not self._client:
            raise RuntimeError("AgentRunner not started — call start() first")

        response = await self._client.post(
            "/v1/messages",
            json={
                "model": self._settings.model,
                "max_tokens": self._settings.max_tokens,
                "system": system_prompt,
                "messages": messages,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    def _get_or_create_conversation(self, session_id: str) -> Conversation:
        """Get existing or create new conversation with LRU eviction."""
        if session_id in self._conversations:
            # Move to end (most recently used)
            self._conversations.move_to_end(session_id)
            return self._conversations[session_id]

        # Evict oldest if at capacity
        while len(self._conversations) >= MAX_CONVERSATIONS:
            self._conversations.popitem(last=False)

        conversation = Conversation(session_id=session_id)
        self._conversations[session_id] = conversation
        return conversation

    def _format_messages(self, conversation: Conversation) -> list[dict]:
        """Format conversation history for Anthropic API.

        Returns list of {"role": ..., "content": ...}.
        Limits to last MAX_HISTORY_MESSAGES messages.
        """
        recent = conversation.messages[-MAX_HISTORY_MESSAGES:]
        return [{"role": m.role, "content": m.content} for m in recent]
