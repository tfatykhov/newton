"""Agent runner -- executes conversational turns via direct Anthropic API.

Wires CognitiveLayer.pre_turn() and post_turn() around
direct httpx calls to the Anthropic Messages API.  Manages the
tool use loop internally (no external SDK).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import httpx

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

# Frame-gated tool access (D5)
FRAME_TOOLS: dict[str, list[str]] = {
    "conversation": ["record_decision", "learn_fact", "recall_deep", "create_censor", "bash", "read_file", "write_file", "web_search", "web_fetch"],
    "question": ["recall_deep", "bash", "read_file", "write_file", "record_decision", "learn_fact", "create_censor", "web_search", "web_fetch"],
    "decision": ["record_decision", "recall_deep", "create_censor", "bash", "read_file", "web_search", "web_fetch"],
    "creative": ["learn_fact", "recall_deep", "write_file", "web_search"],
    "task": ["*"],  # All tools
    "debug": ["record_decision", "recall_deep", "bash", "read_file", "learn_fact", "web_search", "web_fetch"],
}

# Anthropic API version header (P0-1)
_API_VERSION = "2023-06-01"


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


@dataclass
class ApiResponse:
    """Parsed response from Anthropic Messages API."""

    content: list[dict[str, Any]]  # Raw content blocks from API
    stop_reason: str  # end_turn, max_tokens, tool_use, stop_sequence
    usage: dict[str, int] | None = None


@dataclass
class StreamEvent:
    """A single event from the streaming API response."""

    type: str  # text_delta, tool_start, tool_input_delta, tool_end, block_stop, done, error, message_stop
    text: str = ""
    tool_name: str = ""
    tool_id: str = ""
    tool_input: dict = field(default_factory=dict)
    stop_reason: str = ""
    block_index: int = 0


def _parse_sse_event(data: dict[str, Any]) -> StreamEvent | None:
    """Parse Anthropic SSE event dict into StreamEvent.

    N4: Skip ping keepalives.
    N3: stop_reason is in message_delta.delta, NOT message_start.
    N2: Handle in-stream error events (HTTP 200 but error in body).
    """
    event_type = data.get("type")

    if event_type == "ping":
        return None

    if event_type == "error":
        error = data.get("error", {})
        return StreamEvent(
            type="error",
            text=f"{error.get('type', 'unknown')}: {error.get('message', '')}",
        )

    if event_type == "content_block_start":
        block = data.get("content_block", {})
        block_index = data.get("index", 0)
        if block.get("type") == "tool_use":
            return StreamEvent(
                type="tool_start",
                tool_name=block.get("name", ""),
                tool_id=block.get("id", ""),
                block_index=block_index,
            )
        return StreamEvent(type="text_block_start", block_index=block_index)

    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        block_index = data.get("index", 0)
        if delta.get("type") == "text_delta":
            return StreamEvent(type="text_delta", text=delta.get("text", ""))
        if delta.get("type") == "input_json_delta":
            return StreamEvent(
                type="tool_input_delta",
                text=delta.get("partial_json", ""),
                block_index=block_index,
            )
        return None

    if event_type == "content_block_stop":
        return StreamEvent(type="block_stop", block_index=data.get("index", 0))

    if event_type == "message_delta":
        return StreamEvent(
            type="done",
            stop_reason=data.get("delta", {}).get("stop_reason", ""),
        )

    if event_type == "message_stop":
        return StreamEvent(type="message_stop")

    return None


class AgentRunner:
    """Runs conversational turns with cognitive layer hooks.

    Uses direct httpx calls to the Anthropic Messages API with
    an internal tool dispatch loop.
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
        self._http: httpx.AsyncClient | None = None
        self._dispatcher: Any | None = None  # ToolDispatcher, set via set_dispatcher()

    def set_dispatcher(self, dispatcher: Any) -> None:
        """Set the tool dispatcher for tool loop execution."""
        self._dispatcher = dispatcher

    async def start(self) -> None:
        """Initialize the httpx client with auth and timeout settings."""
        settings = self._settings

        # Build default headers
        headers: dict[str, str] = {
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        # Auth header selection (D1 + OAT detection)
        # OAT tokens (sk-ant-oat*) from `claude setup-token` require Bearer auth
        # plus special beta headers. Regular API keys use x-api-key.
        api_key = settings.anthropic_api_key or ""
        auth_token = settings.anthropic_auth_token or ""

        if auth_token:
            # Explicit auth token always uses Bearer
            headers["authorization"] = f"Bearer {auth_token}"
            if "sk-ant-oat" in auth_token:
                headers["anthropic-beta"] = "oauth-2025-04-20"
                headers["anthropic-dangerous-direct-browser-access"] = "true"
        elif api_key:
            if "sk-ant-oat" in api_key:
                # OAT token passed as API key - use Bearer + required beta headers
                headers["authorization"] = f"Bearer {api_key}"
                headers["anthropic-beta"] = "oauth-2025-04-20"
                headers["anthropic-dangerous-direct-browser-access"] = "true"
            else:
                headers["x-api-key"] = api_key
        else:
            logger.warning(
                "Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set -- "
                "API calls will fail"
            )

        # Create httpx client with timeout and connection limits
        timeout = httpx.Timeout(
            connect=settings.api_timeout_connect,
            read=settings.api_timeout_read,
            write=10.0,
            pool=10.0,
        )
        limits = httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
        )

        self._http = httpx.AsyncClient(
            base_url=settings.api_base_url,
            headers=headers,
            timeout=timeout,
            limits=limits,
        )

        is_oat = "sk-ant-oat" in (auth_token or api_key)
        auth_type = "OAT/subscription" if is_oat else ("Bearer token" if auth_token else "API key")
        logger.info("httpx client initialized (auth: %s)", auth_type)

    async def close(self) -> None:
        """Clean up httpx client."""
        if self._http:
            await self._http.aclose()
            self._http = None

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
        4. Build system prompt (cognitive context + frame instructions)
        5. Run tool loop: call API, dispatch tools, repeat until done
        6. Extract response text from final API response
        7. Append assistant message to conversation history
        8. Call cognitive.post_turn() with TurnResult
        9. Safety net: check if decision frame but no record_decision called
        10. Return (response_text, turn_context)
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

        # 4-6. Build system prompt and run tool loop
        response_text = ""
        tool_results: list[ToolResult] = []
        error = None
        try:
            system_prompt = self._build_system_prompt(turn_context)
            response_text, tool_results = await self._tool_loop(
                system_prompt=system_prompt,
                conversation=conversation,
                frame_id=turn_context.frame.frame_id,
            )
            conversation.messages.append(Message(role="assistant", content=response_text))
        except Exception as e:
            logger.error("API call error: %s", e)
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
                # P1-9: Call _call_api directly, no tool loop needed for reflection
                api_response = await self._call_api(
                    system_prompt="You are reviewing a conversation to extract lessons learned.",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    tools=None,
                )
                # Extract text from response
                reflection = self._extract_text(api_response.content)
            except Exception as e:
                logger.warning("Failed to generate reflection: %s", e)

        await self._cognitive.end_session(_agent_id, session_id, reflection=reflection)

        # Remove conversation
        self._conversations.pop(session_id, None)

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _build_api_payload(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build Anthropic Messages API request payload.

        Shared by _call_api and _call_api_stream to avoid divergence.
        """
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "max_tokens": self._settings.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return payload

    async def _call_api(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ApiResponse:
        """Call Anthropic Messages API with retry for 429/500/529.

        Returns parsed ApiResponse with content blocks and stop_reason.
        Raises RuntimeError on persistent errors.
        """
        if not self._http:
            raise RuntimeError("httpx client not initialized -- call start() first")

        payload = self._build_api_payload(system_prompt, messages, tools)

        # Simple retry: 1x for 429/500/529
        last_error: Exception | None = None
        for attempt in range(2):  # max 2 attempts (initial + 1 retry)
            try:
                response = await self._http.post("/v1/messages", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    return ApiResponse(
                        content=data["content"],
                        stop_reason=data["stop_reason"],
                        usage=data.get("usage"),
                    )

                # Parse error body
                try:
                    error_data = response.json()
                    error_type = error_data.get("error", {}).get("type", "unknown")
                    error_msg = error_data.get("error", {}).get("message", "unknown error")
                except Exception:
                    error_type = "http_error"
                    error_msg = f"HTTP {response.status_code}: {response.text[:500]}"

                # Retry on 429 (rate limit) or 500/529 (server error)
                if response.status_code in (429, 500, 529) and attempt == 0:
                    retry_after = float(response.headers.get("retry-after", "1"))
                    retry_after = min(retry_after, 30.0)  # Cap at 30s
                    logger.warning(
                        "API error %d (%s), retrying in %.1fs: %s",
                        response.status_code,
                        error_type,
                        retry_after,
                        error_msg,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                last_error = RuntimeError(
                    f"Anthropic API error ({response.status_code}): "
                    f"{error_type} - {error_msg}"
                )

            except httpx.TimeoutException as e:
                last_error = RuntimeError(f"API request timed out: {e}")
                if attempt == 0:
                    logger.warning("API timeout, retrying: %s", e)
                    await asyncio.sleep(1)
                    continue
            except httpx.HTTPError as e:
                last_error = RuntimeError(f"HTTP error: {e}")
                break  # Don't retry connection errors

        raise last_error or RuntimeError("API call failed with unknown error")

    async def _call_api_stream(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Call Anthropic API with streaming enabled.

        Yields StreamEvent objects. Uses self._http which already has
        auth headers and base_url configured (set in start()).

        N2: Yields error event on HTTP errors or in-stream errors.
        N8: Only processes data: lines (skips event: lines naturally).
        """
        if not self._http:
            raise RuntimeError("httpx client not initialized -- call start() first")

        payload = self._build_api_payload(system_prompt, messages, tools, stream=True)

        async with self._http.stream("POST", "/v1/messages", json=payload) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield StreamEvent(type="error", text=error_body.decode()[:500])
                return

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                event = _parse_sse_event(data)
                if event:
                    if event.type == "error":
                        yield event
                        return
                    yield event

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        agent_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Full chat turn with streaming, including tool loops.

        Mirrors run_turn() flow but yields StreamEvents as they arrive.
        Tool calls execute between stream segments.
        """
        if not self._dispatcher:
            raise RuntimeError("No tool dispatcher set -- call set_dispatcher() first")

        _agent_id = agent_id or self._settings.agent_id
        conversation = self._get_or_create_conversation(session_id)

        # Pre-turn with conversation dedup (F4)
        recent_messages = [
            m.content for m in conversation.messages if m.role == "user"
        ][-8:]
        turn_context = await self._cognitive.pre_turn(
            _agent_id,
            session_id,
            user_message,
            conversation_messages=recent_messages or None,
        )

        conversation.messages.append(Message(role="user", content=user_message))

        system_prompt = self._build_system_prompt(turn_context)
        tools = self._dispatcher.available_tools(turn_context.frame.frame_id)
        messages = self._format_messages(conversation)

        all_tool_results: list[ToolResult] = []
        response_text = ""
        error = None

        try:
            for turn in range(self._settings.max_turns):
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                block_accumulators: dict[int, dict[str, Any]] = {}
                stop_reason = ""

                async for event in self._call_api_stream(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools if tools else None,
                ):
                    if event.type == "error":
                        error = event.text
                        yield event
                        return

                    elif event.type == "text_delta":
                        text_parts.append(event.text)
                        yield event

                    elif event.type == "tool_start":
                        block_accumulators[event.block_index] = {
                            "id": event.tool_id,
                            "name": event.tool_name,
                            "input_parts": [],
                        }
                        yield event

                    elif event.type == "tool_input_delta":
                        acc = block_accumulators.get(event.block_index)
                        if acc:
                            acc["input_parts"].append(event.text)

                    elif event.type == "block_stop":
                        acc = block_accumulators.pop(event.block_index, None)
                        if acc:
                            input_json = "".join(acc["input_parts"])
                            try:
                                acc["input"] = json.loads(input_json) if input_json else {}
                            except json.JSONDecodeError:
                                acc["input"] = {}
                            tool_calls.append(acc)

                    elif event.type == "done":
                        stop_reason = event.stop_reason

                # Stream segment ended -- decide next action
                if stop_reason == "end_turn" or not tool_calls:
                    response_text = "".join(text_parts)
                    break

                # Build assistant message with tool_use content blocks
                content_blocks: list[dict[str, Any]] = []
                if text_parts:
                    content_blocks.append({"type": "text", "text": "".join(text_parts)})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    })
                messages.append({"role": "assistant", "content": content_blocks})

                # Execute tools (P1-2: all results in single user message)
                tool_results_for_message: list[dict[str, Any]] = []
                for tc in tool_calls:
                    yield StreamEvent(type="tool_start", tool_name=tc["name"])
                    start_time = time.monotonic()
                    try:
                        result_text, is_error = await self._dispatcher.dispatch(
                            tc["name"], tc["input"]
                        )
                    except Exception as e:
                        result_text = str(e)
                        is_error = True
                    duration_ms = int((time.monotonic() - start_time) * 1000)

                    tool_results_for_message.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result_text,
                        "is_error": is_error,
                    })

                    all_tool_results.append(ToolResult(
                        tool_name=tc["name"],
                        arguments=tc["input"],
                        result=result_text if not is_error else None,
                        error=result_text if is_error else None,
                        duration_ms=duration_ms,
                    ))

                    yield StreamEvent(type="tool_end", tool_name=tc["name"])

                messages.append({"role": "user", "content": tool_results_for_message})
            else:
                # Max turns reached -- final call without tools
                logger.warning("Streaming tool loop reached max_turns=%d", self._settings.max_turns)
                try:
                    final = await self._call_api(
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=None,
                    )
                    response_text = self._extract_text(final.content)
                except Exception:
                    response_text = "I reached the maximum number of tool iterations."

            # Store assistant response
            conversation.messages.append(Message(role="assistant", content=response_text))

        except Exception as e:
            logger.error("Streaming error: %s", e)
            error = str(e)
            response_text = "I encountered an error processing your request."
            conversation.messages.append(Message(role="assistant", content=response_text))
        finally:
            # ALWAYS call post_turn (review P1: guaranteed cleanup)
            turn_result = TurnResult(
                response_text=response_text,
                tool_results=all_tool_results,
                error=error,
            )
            await self._cognitive.post_turn(_agent_id, session_id, turn_result, turn_context)
            self._check_safety_net(turn_context, all_tool_results)
            conversation.turn_contexts.append(turn_context)

        yield StreamEvent(type="done", stop_reason="end_turn")

    # ------------------------------------------------------------------
    # Tool loop
    # ------------------------------------------------------------------

    async def _tool_loop(
        self,
        system_prompt: str,
        conversation: Conversation,
        frame_id: str,
    ) -> tuple[str, list[ToolResult]]:
        """Run the tool use loop until completion or max_turns.

        Returns (response_text, tool_results).

        The loop:
        1. Build messages array from conversation
        2. Call API with system prompt, messages, and available tools
        3. If stop_reason is not "tool_use", extract text and return
        4. Otherwise: append assistant response, dispatch tools, append results
        5. Repeat until done or max_turns
        """
        if not self._dispatcher:
            raise RuntimeError("No tool dispatcher set -- call set_dispatcher() first")

        # Get tools for current frame (D5)
        tools = self._dispatcher.available_tools(frame_id)

        # Build initial messages from conversation history
        # The latest user message is already in conversation.messages
        messages = self._format_messages(conversation)

        all_tool_results: list[ToolResult] = []
        turns = 0
        max_turns = self._settings.max_turns

        while turns < max_turns:
            api_response = await self._call_api(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools if tools else None,
            )

            # If not a tool use, we're done
            if api_response.stop_reason != "tool_use":
                response_text = self._extract_text(api_response.content)
                return response_text, all_tool_results

            # P0-3: Append FULL assistant response (all content blocks)
            messages.append({
                "role": "assistant",
                "content": api_response.content,
            })

            # P1-2: Dispatch ALL tool_use blocks, collect results in SINGLE user message
            tool_results_for_message: list[dict[str, Any]] = []
            for block in api_response.content:
                if block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_input = block.get("input", {})
                    tool_use_id = block["id"]

                    start_time = time.monotonic()
                    result_text, is_error = await self._dispatcher.dispatch(
                        tool_name, tool_input
                    )
                    duration_ms = int((time.monotonic() - start_time) * 1000)

                    tool_results_for_message.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_text,
                        "is_error": is_error,
                    })

                    # Track for post_turn
                    all_tool_results.append(ToolResult(
                        tool_name=tool_name,
                        arguments=tool_input,
                        result=result_text if not is_error else None,
                        error=result_text if is_error else None,
                        duration_ms=duration_ms,
                    ))

            # Append all tool results as single user message
            messages.append({
                "role": "user",
                "content": tool_results_for_message,
            })

            turns += 1

        # Max turns reached -- extract text from last response if any
        logger.warning("Tool loop reached max_turns=%d", max_turns)
        # Make one final call without tools to get a text response
        try:
            final_response = await self._call_api(
                system_prompt=system_prompt,
                messages=messages,
                tools=None,
            )
            return self._extract_text(final_response.content), all_tool_results
        except Exception:
            return "I reached the maximum number of tool iterations. Please try again.", all_tool_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        turn_context: TurnContext,
    ) -> str:
        """Build the system prompt: cognitive context + frame instructions.

        P0-2 fix: Does NOT inject conversation history. History flows
        through messages[] array via _format_messages().
        """
        parts = [turn_context.system_prompt]

        # Add frame-specific tool instructions
        frame_instructions = self._get_frame_instructions(turn_context)
        if frame_instructions:
            parts.append(frame_instructions)

        return "\n\n".join(parts)

    def _get_frame_instructions(self, turn_context: TurnContext) -> str:
        """Return frame-specific tool use instructions.

        Nudges Claude to use appropriate tools based on the active cognitive frame.
        Only mentions tools available for the frame (synced with FRAME_TOOLS).
        """
        frame_id = turn_context.frame.frame_id

        if frame_id == "decision":
            return (
                "## Tool Instructions\n\n"
                "You are in a DECISION frame. You MUST call `record_decision` "
                "to record your decision before responding. Include your reasoning, "
                "confidence level, and category.\n\n"
                "**What IS a decision:** A choice between alternatives — architecture "
                "choices, tool selections, process changes, trade-offs with pros/cons.\n\n"
                "**What is NOT a decision:** Status reports, routine completions, "
                "simple observations, task acknowledgments, greetings. "
                "Do NOT record these.\n\n"
                "Use `recall_deep` to search for relevant past decisions. "
                "Use `web_search` and `web_fetch` to research options before deciding."
            )
        elif frame_id == "task":
            return (
                "## Tool Instructions\n\n"
                "You are in a TASK frame. If you make a meaningful choice between "
                "alternatives during this task, call `record_decision` to record it. "
                "Do NOT record routine task completions, status updates, or simple "
                "observations as decisions — a decision requires choosing between "
                "alternatives with trade-offs.\n\n"
                "Use `recall_deep` to search for relevant past decisions and knowledge. "
                "Use `learn_fact` to store any new facts discovered. You can also use "
                "`bash`, `read_file`, and `write_file` for system operations. "
                "Use `web_search` and `web_fetch` for research."
            )
        elif frame_id == "debug":
            return (
                "## Tool Instructions\n\n"
                "You are in a DEBUG frame. Use `recall_deep` to search for relevant "
                "past decisions and procedures. Record meaningful debugging decisions "
                "(e.g., root cause identified, fix approach chosen) with "
                "`record_decision`. Do NOT record routine debug steps or status "
                "observations. Store root cause findings with `learn_fact`. "
                "Use `bash` and `read_file` for investigation. Use `web_search` and "
                "`web_fetch` to look up documentation or error messages."
            )
        elif frame_id == "question":
            return (
                "## Tool Instructions\n\n"
                "You are in a QUESTION frame. Use `recall_deep` to search memory "
                "for relevant knowledge before answering. Use `web_search` and "
                "`web_fetch` for questions about current events or topics not in memory."
            )
        elif frame_id == "creative":
            return (
                "## Tool Instructions\n\n"
                "You are in a CREATIVE frame. Use `recall_deep` to find relevant "
                "knowledge. Use `learn_fact` to store creative insights. "
                "Use `write_file` to save creative output. Use `web_search` "
                "for inspiration and reference material."
            )
        elif frame_id == "conversation":
            return (
                "## Tool Instructions\n\n"
                "You are in a CONVERSATION frame. Use `web_search` and `web_fetch` "
                "to find current information when needed."
            )

        return ""

    def _extract_text(self, content_blocks: list[dict[str, Any]]) -> str:
        """Extract text from API response content blocks."""
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block["text"])
        return "\n".join(text_parts) if text_parts else ""

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

    def _format_messages(self, conversation: Conversation) -> list[dict[str, Any]]:
        """Format conversation history for API calls.

        Returns list of {"role": ..., "content": ...}.
        Limits to last MAX_HISTORY_MESSAGES messages.
        """
        recent = conversation.messages[-MAX_HISTORY_MESSAGES:]
        return [{"role": m.role, "content": m.content} for m in recent]

    def _format_history_text(self, conversation: Conversation) -> str:
        """Format conversation history as readable text for reflection."""
        recent = conversation.messages[-MAX_HISTORY_MESSAGES:]
        lines = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        return "\n\n".join(lines)
