"""Agent runner — executes a single conversational turn.

Wires CognitiveLayer.pre_turn() and post_turn() around
the Claude Agent SDK query. This is the core execution loop.

Uses claude-agent-sdk with in-process MCP tools so Claude can
call record_decision, learn_fact, recall_deep, and create_censor
during a turn.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from nous.brain.brain import Brain
from nous.cognitive.layer import CognitiveLayer
from nous.cognitive.schemas import ToolResult, TurnContext, TurnResult
from nous.config import Settings
from nous.heart.heart import Heart

logger = logging.getLogger(__name__)

MAX_CONVERSATIONS = 100
MAX_HISTORY_MESSAGES = 20

# Frame types that should nudge Claude to call record_decision
_DECISION_FRAMES = frozenset({"decision", "task", "debug"})


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

    Uses the Claude Agent SDK with in-process MCP tools for
    tool-augmented LLM conversations.
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
        brain: Brain,
        heart: Heart,
        settings: Settings,
    ) -> None:
        self._cognitive = cognitive
        self._brain = brain
        self._heart = heart
        self._settings = settings
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()
        self._mcp_server: Any | None = None  # McpSdkServerConfig from tools.py

    async def start(self) -> None:
        """Initialize the SDK: set auth env vars and create in-process MCP server."""
        # Set Anthropic auth environment variables for the SDK.
        # auth_token takes precedence over api_key.
        if self._settings.anthropic_auth_token:
            os.environ["ANTHROPIC_AUTH_TOKEN"] = self._settings.anthropic_auth_token
        elif self._settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self._settings.anthropic_api_key
        else:
            logger.warning(
                "Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set — "
                "SDK queries will fail"
            )

        # Create in-process MCP server with Nous tools
        from nous.api.tools import create_nous_mcp_server

        self._mcp_server = create_nous_mcp_server(self._brain, self._heart)
        logger.info(
            "SDK tools registered: record_decision, learn_fact, recall_deep, create_censor"
        )

    async def close(self) -> None:
        """Clean up SDK resources."""
        self._mcp_server = None

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
        4. Build full system prompt (cognitive context + frame instructions + history)
        5. Call Claude via SDK query() with in-process tools
        6. Extract response text and tool call info from stream
        7. Append assistant message to conversation history
        8. Call cognitive.post_turn() with TurnResult
        9. Safety net: check if decision frame but no record_decision called
        10. Return (response_text, turn_context)

        On SDK error: log, create TurnResult with error, still call post_turn.
        """
        _agent_id = agent_id or self._settings.agent_id
        conversation = self._get_or_create_conversation(session_id)

        # 2. Pre-turn (F4: plumb conversation_messages for dedup)
        # Filter to user messages first, then take last 8 (D7: window = user turns)
        recent_messages = [
            m.content for m in conversation.messages if m.role == "user"
        ][-8:]
        turn_context = await self._cognitive.pre_turn(
            _agent_id,
            session_id,
            user_message,
            conversation_messages=recent_messages or None,
        )

        # 3. Append user message
        conversation.messages.append(Message(role="user", content=user_message))

        # 4-6. Call LLM via SDK
        response_text = ""
        tool_results: list[ToolResult] = []
        error = None
        try:
            response_text, tool_results = await self._call_sdk(
                system_prompt=self._build_system_prompt(turn_context, conversation),
                user_message=user_message,
            )
            conversation.messages.append(Message(role="assistant", content=response_text))
        except Exception as e:
            logger.error("SDK query error: %s", e)
            error = str(e)
            response_text = "I encountered an error processing your request. Please try again."
            conversation.messages.append(Message(role="assistant", content=response_text))

        # 7. Post-turn (always called, even on error)
        turn_result = TurnResult(
            response_text=response_text,
            tool_results=tool_results,
            error=error,
        )
        await self._cognitive.post_turn(_agent_id, session_id, turn_result, turn_context)

        # 8. Safety net: warn if decision frame but record_decision not called
        self._check_safety_net(turn_context, tool_results)

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
        if conversation and len(conversation.messages) >= 6:  # 3 user + 3 assistant = 6
            try:
                # Build a reflection prompt with conversation history
                history_text = self._format_history_text(conversation)
                reflection_prompt = (
                    f"Here is a conversation to review:\n\n{history_text}\n\n"
                    f"{self.REFLECTION_PROMPT}"
                )
                reflection, _ = await self._call_sdk(
                    system_prompt="You are reviewing a conversation to extract lessons learned.",
                    user_message=reflection_prompt,
                )
            except Exception as e:
                logger.warning("Failed to generate reflection: %s", e)

        await self._cognitive.end_session(_agent_id, session_id, reflection=reflection)

        # Remove conversation
        self._conversations.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_sdk(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, list[ToolResult]]:
        """Call Claude via the SDK query() function.

        Returns (response_text, tool_results).
        The SDK handles the tool loop internally — we iterate the stream
        and collect the final text response and any tool call info.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self._settings.model,
            max_turns=self._settings.sdk_max_turns,
            permission_mode=self._settings.sdk_permission_mode,
        )

        # Register in-process MCP server if available
        if self._mcp_server:
            options.mcp_servers = {"nous": self._mcp_server}

        response_text_parts: list[str] = []
        tool_results: list[ToolResult] = []
        # Map tool_use_id -> index in tool_results for attaching results
        _tool_use_index: dict[str, int] = {}

        async for message in query(prompt=user_message, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        _tool_use_index[block.id] = len(tool_results)
                        tool_results.append(
                            ToolResult(
                                tool_name=block.name,
                                arguments=block.input,
                            )
                        )
                    elif isinstance(block, ToolResultBlock):
                        idx = _tool_use_index.get(block.tool_use_id)
                        if idx is not None:
                            content = block.content
                            if isinstance(content, list):
                                content = "\n".join(
                                    item.get("text", "") for item in content if isinstance(item, dict)
                                )
                            tool_results[idx].result = content or None
                            if block.is_error:
                                tool_results[idx].error = content or "tool error"
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(
                        f"SDK query failed (session={message.session_id}): "
                        f"{message.result or 'unknown error'}"
                    )
                # If ResultMessage has a result text and we got no text blocks,
                # use it as the response
                if message.result and not response_text_parts:
                    response_text_parts.append(message.result)

        response_text = "\n".join(response_text_parts) if response_text_parts else ""
        return response_text, tool_results

    def _build_system_prompt(
        self,
        turn_context: TurnContext,
        conversation: Conversation,
    ) -> str:
        """Build the full system prompt: cognitive context + frame instructions + history."""
        parts = [turn_context.system_prompt]

        # Add frame-specific tool instructions
        frame_instructions = self._get_frame_instructions(turn_context)
        if frame_instructions:
            parts.append(frame_instructions)

        # Add conversation history as context
        if conversation.messages:
            history = self._format_history_text(conversation)
            if history:
                parts.append(
                    f"## Conversation History\n\n{history}\n\n"
                    "Continue the conversation from here."
                )

        return "\n\n".join(parts)

    def _get_frame_instructions(self, turn_context: TurnContext) -> str:
        """Return frame-specific tool use instructions.

        Nudges Claude to use appropriate tools based on the active cognitive frame.
        """
        frame_id = turn_context.frame.frame_id

        if frame_id == "decision":
            return (
                "## Tool Instructions\n\n"
                "You are in a DECISION frame. You MUST call `record_decision` "
                "to record your decision before responding. Include your reasoning, "
                "confidence level, and category."
            )
        elif frame_id == "task":
            return (
                "## Tool Instructions\n\n"
                "You are in a TASK frame. If you make any decisions during this task, "
                "call `record_decision` to record them. Use `recall_deep` to search "
                "for relevant past decisions and knowledge. Use `learn_fact` to store "
                "any new facts discovered."
            )
        elif frame_id == "debug":
            return (
                "## Tool Instructions\n\n"
                "You are in a DEBUG frame. Use `recall_deep` to search for relevant "
                "past decisions and procedures. Record your debugging decisions with "
                "`record_decision`. Store any root cause findings with `learn_fact`."
            )
        elif frame_id == "question":
            return (
                "## Tool Instructions\n\n"
                "You are in a QUESTION frame. Use `recall_deep` to search memory "
                "for relevant knowledge before answering."
            )

        return ""

    def _check_safety_net(
        self,
        turn_context: TurnContext,
        tool_results: list[ToolResult],
    ) -> None:
        """Log a warning if a decision frame was active but record_decision wasn't called."""
        frame_id = turn_context.frame.frame_id
        if frame_id not in _DECISION_FRAMES:
            return

        tool_names = {tr.tool_name for tr in tool_results}
        if "record_decision" not in tool_names:
            logger.warning(
                "Safety net: frame=%s but record_decision was not called during turn "
                "(session decision_id=%s). Consider recording this decision.",
                frame_id,
                turn_context.decision_id,
            )

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
        """Format conversation history for API calls.

        Returns list of {"role": ..., "content": ...}.
        Limits to last MAX_HISTORY_MESSAGES messages.
        """
        recent = conversation.messages[-MAX_HISTORY_MESSAGES:]
        return [{"role": m.role, "content": m.content} for m in recent]

    def _format_history_text(self, conversation: Conversation) -> str:
        """Format conversation history as readable text for system prompt injection."""
        recent = conversation.messages[-MAX_HISTORY_MESSAGES:]
        lines = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        return "\n\n".join(lines)
