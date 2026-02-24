"""Telegram bot for Nous agent.

Polls Telegram for messages and forwards them to the Nous /chat REST API.
Sends responses back to the user.

Usage:
    TELEGRAM_BOT_TOKEN=... NOUS_API_URL=http://localhost:8000 python -m nous.telegram_bot

Environment:
    TELEGRAM_BOT_TOKEN  - Bot token from @BotFather
    NOUS_API_URL        - Nous REST API base URL (default: http://localhost:8000)
    NOUS_ALLOWED_USERS  - Comma-separated Telegram user IDs (optional, empty = allow all)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Telegram Bot API base
TG_API = "https://api.telegram.org/bot{token}/{method}"

# Max Telegram message length
TG_MAX_LEN = 4096


def format_usage_footer(usage: dict[str, int]) -> str:
    """Format token usage as a compact footer string."""
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    inp_str = f"{inp / 1000:.1f}K" if inp >= 1000 else str(inp)
    out_str = f"{out / 1000:.1f}K" if out >= 1000 else str(out)
    return f"\U0001f4ca {inp_str} in / {out_str} out"


class StreamingMessage:
    """Manages progressive message editing for Telegram streaming."""

    # Tool name -> (emoji, display label)
    TOOL_INDICATORS: dict[str, tuple[str, str]] = {
        "web_search": ("\U0001f50d", "Searching"),
        "web_fetch": ("\U0001f310", "Fetching"),
        "recall_deep": ("\U0001f9e0", "Remembering"),
        "record_decision": ("\U0001f4dd", "Recording"),
        "learn_fact": ("\U0001f4a1", "Learning"),
        "bash": ("\u2699\ufe0f", "Running"),
        "read_file": ("\U0001f4c4", "Reading"),
        "write_file": ("\u270f\ufe0f", "Writing"),
    }

    def __init__(self, bot: NousTelegramBot, chat_id: int):
        self._bot = bot
        self.chat_id = chat_id
        self.message_id: int | None = None
        self.text = ""
        self._base_text = ""  # Text without tool indicators
        self._tool_counts: dict[str, int] = {}  # tool_name -> count
        self._last_edit = 0.0
        self._min_interval = 1.2  # N6: ~20 edits/msg limit
        self._pending = False
        self._usage: dict[str, int] | None = None  # Token usage stats

    def set_usage(self, usage: dict[str, int]) -> None:
        """Set token usage for the footer."""
        self._usage = usage

    async def append_text(self, text: str) -> None:
        """Append text delta to the base text and update display."""
        await self.update(self._base_text + text)

    async def update(self, new_text: str) -> None:
        """Update message text. Creates on first call, edits after."""
        self._base_text = new_text
        self.text = self._build_display_text()
        now = time.time()
        if now - self._last_edit < self._min_interval:
            self._pending = True
            return
        await self._send_or_edit()

    async def append_tool_indicator(self, tool_name: str) -> None:
        """Track tool usage with grouped counters instead of appending lines."""
        self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1
        self.text = self._build_display_text()
        await self._send_or_edit()

    def _build_display_text(self) -> str:
        """Build display text: base text + collapsed tool indicators + usage footer."""
        parts = [self._base_text] if self._base_text else []

        if self._tool_counts:
            indicators: list[str] = []
            for tool_name, count in self._tool_counts.items():
                emoji, label = self.TOOL_INDICATORS.get(
                    tool_name, ("\U0001f527", tool_name)
                )
                if count > 1:
                    indicators.append(f"{emoji} {label}... ({count})")
                else:
                    indicators.append(f"{emoji} {label}...")
            parts.append(" ".join(indicators))

        return "\n\n".join(parts)

    async def finalize(self) -> None:
        """Send final version of message with usage footer."""
        # Clear tool indicators for final message, keep only base text + footer
        self._tool_counts.clear()
        self.text = self._build_display_text()

        # Append usage footer if available
        if self._usage:
            self.text += f"\n\n{format_usage_footer(self._usage)}"

        await self._send_or_edit()

    async def _send_or_edit(self) -> None:
        if not self.text.strip():
            return

        display_text = self.text

        # N7: Handle 4096 char overflow
        if len(display_text) > 4000 and self.message_id is not None:
            overflow = display_text[4000:]
            truncated = display_text[:4000] + "\n\n(continued...)"
            await self._bot._tg("editMessageText", params={
                "chat_id": self.chat_id,
                "message_id": self.message_id,
                "text": truncated,
            })
            result = await self._bot._send(self.chat_id, overflow)
            if isinstance(result, dict) and "message_id" in result:
                self.message_id = result["message_id"]
            self.text = overflow
            self._last_edit = time.time()
            self._pending = False
            return

        if self.message_id is None:
            result = await self._bot._send(self.chat_id, display_text)
            if isinstance(result, dict) and "message_id" in result:
                self.message_id = result["message_id"]
        else:
            # No parse_mode during streaming (review B3: partial markdown breaks)
            await self._bot._tg("editMessageText", params={
                "chat_id": self.chat_id,
                "message_id": self.message_id,
                "text": display_text,
            })
        self._last_edit = time.time()
        self._pending = False


class NousTelegramBot:
    """Lightweight Telegram bot that proxies to Nous /chat API."""

    def __init__(
        self,
        bot_token: str,
        nous_url: str = "http://localhost:8000",
        allowed_users: set[int] | None = None,
    ):
        self.bot_token = bot_token
        self.nous_url = nous_url.rstrip("/")
        self.allowed_users = allowed_users
        self._offset = 0
        # Map telegram chat_id -> nous session_id for continuity
        self._sessions: dict[int, str] = {}
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10))

    async def start(self) -> None:
        """Start polling loop."""
        me = await self._tg("getMe")
        logger.info("Bot started: @%s (%s)", me.get("username"), me.get("id"))
        print(f"Nous Telegram bot started: @{me.get('username')}")

        while True:
            try:
                updates = await self._tg(
                    "getUpdates",
                    params={"offset": self._offset, "timeout": 30},
                )
                for update in updates:
                    self._offset = update["update_id"] + 1
                    await self._handle_update(update)
            except httpx.ReadTimeout:
                continue  # Normal for long polling
            except Exception as e:
                logger.error("Polling error: %s", e)
                await asyncio.sleep(5)

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """Handle a single Telegram update."""
        message = update.get("message")
        if not message:
            return

        chat_id = message["chat"]["id"]
        user_id = message.get("from", {}).get("id")
        text = message.get("text", "").strip()

        if not text:
            return

        # Access control
        if self.allowed_users and user_id not in self.allowed_users:
            await self._send(chat_id, "⛔ Not authorized.")
            return

        # Handle /start
        if text == "/start":
            await self._send(chat_id, "🧠 *Nous* is ready\\. Send me a message\\!", parse_mode="MarkdownV2")
            return

        # Handle /new - reset session
        if text == "/new":
            self._sessions.pop(chat_id, None)
            await self._send(chat_id, "🔄 New session started.")
            return

        # Handle /debug - toggle debug mode
        if text == "/debug":
            await self._chat(chat_id, "Show me your current status and available tools.", debug=True)
            return

        # Forward to Nous (streaming)
        await self._chat_streaming(chat_id, text)

    async def _chat(self, chat_id: int, text: str, debug: bool = False) -> None:
        """Send message to Nous API and relay response to Telegram."""
        # Send typing indicator
        await self._tg("sendChatAction", params={"chat_id": chat_id, "action": "typing"})

        session_id = self._sessions.get(chat_id)
        payload: dict[str, Any] = {"message": text}
        if session_id:
            payload["session_id"] = session_id
        if debug:
            payload["debug"] = True

        try:
            response = await self._http.post(
                f"{self.nous_url}/chat",
                json=payload,
                timeout=120,
            )
            data = response.json()

            if response.status_code != 200:
                await self._send(chat_id, f"❌ Error: {data.get('error', 'Unknown error')}")
                return

            # Store session for continuity
            if "session_id" in data:
                self._sessions[chat_id] = data["session_id"]

            # Build response text
            reply = data.get("response", "No response")
            frame = data.get("frame", "unknown")

            # Add frame indicator
            frame_emoji = {
                "task": "🔧",
                "question": "❓",
                "decision": "⚖️",
                "creative": "🎨",
                "conversation": "💬",
                "debug": "🐛",
            }
            frame_tag = f"{frame_emoji.get(frame, '🧠')} [{frame}]"

            # Add debug info if requested
            if debug and "debug" in data:
                d = data["debug"]
                prompt = d.get("system_prompt", "(empty)")
                debug_text = (
                    f"\n\n---\n🔍 Debug Info:\n"
                    f"Frame: {frame} (confidence: {d.get('frame_confidence', '?')})\n"
                    f"Censors: {d.get('active_censors', 0)}\n"
                    f"Decisions: {d.get('related_decisions', 0)}\n"
                    f"Facts: {d.get('related_facts', 0)}\n"
                    f"Episodes: {d.get('related_episodes', 0)}\n"
                    f"Context tokens: {d.get('context_tokens', 0)}\n"
                    f"\n📋 SYSTEM PROMPT:\n{prompt}"
                )
                reply += debug_text

            # Add usage footer if available
            usage = data.get("usage")
            if usage:
                reply += f"\n\n{format_usage_footer(usage)}"

            # Split long messages
            full_reply = f"{frame_tag}\n\n{reply}"
            await self._send_long(chat_id, full_reply)

            # If decision was recorded, add a subtle indicator
            if data.get("decision_id"):
                await self._tg(
                    "setMessageReaction",
                    params={
                        "chat_id": chat_id,
                        "message_id": (await self._get_last_bot_message_id(chat_id)),
                        "reaction": json.dumps([{"type": "emoji", "emoji": "🧠"}]),
                    },
                )

        except httpx.TimeoutException:
            await self._send(chat_id, "⏱ Request timed out. Nous might be thinking hard...")
        except Exception as e:
            logger.error("Chat error: %s", e)
            await self._send(chat_id, f"❌ Error: {e}")

    async def _chat_streaming(self, chat_id: int, text: str) -> None:
        """Send message to Nous streaming API and progressively edit Telegram message."""
        from uuid import uuid4

        await self._tg("sendChatAction", params={"chat_id": chat_id, "action": "typing"})

        # P1 fix: generate session_id upfront so client and server use same ID
        session_id = self._sessions.setdefault(chat_id, str(uuid4()))
        payload: dict[str, Any] = {"message": text, "session_id": session_id}

        streamer = StreamingMessage(self, chat_id)

        try:
            async with self._http.stream(
                "POST",
                f"{self.nous_url}/chat/stream",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    await self._send(chat_id, f"\u274c Error: {error_body.decode()[:200]}")
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])

                    if event.get("type") == "text_delta":
                        await streamer.append_text(event.get("text", ""))
                    elif event.get("type") == "tool_start":
                        await streamer.append_tool_indicator(event.get("tool_name", ""))
                    elif event.get("type") == "error":
                        await self._send(chat_id, f"\u274c {event.get('text', 'Unknown error')}")
                        return
                    elif event.get("type") == "done":
                        if event.get("usage"):
                            streamer.set_usage(event["usage"])
                        break

        except httpx.TimeoutException:
            await self._send(chat_id, "\u23f1 Request timed out.")
        except Exception as e:
            logger.error("Streaming chat error: %s", e)
            await self._send(chat_id, f"\u274c Error: {e}")
        finally:
            await streamer.finalize()

    async def _send(self, chat_id: int, text: str, parse_mode: str | None = None) -> dict:
        """Send a message to Telegram."""
        params: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            params["parse_mode"] = parse_mode
        return await self._tg("sendMessage", params=params)

    async def _send_long(self, chat_id: int, text: str) -> None:
        """Send a long message, splitting if needed."""
        if len(text) <= TG_MAX_LEN:
            await self._send(chat_id, text)
            return

        # Split on newlines, respecting max length
        chunks: list[str] = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > TG_MAX_LEN:
                if current:
                    chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)

        for chunk in chunks:
            await self._send(chat_id, chunk)
            await asyncio.sleep(0.3)  # Rate limit

    async def _get_last_bot_message_id(self, chat_id: int) -> int | None:
        """Best-effort: return None if we can't track it."""
        return None  # TODO: track sent message IDs

    async def _tg(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Call Telegram Bot API."""
        url = TG_API.format(token=self.bot_token, method=method)
        response = await self._http.get(url, params=params)
        data = response.json()
        if not data.get("ok"):
            logger.warning("Telegram API error: %s", data)
            return data.get("result", [])
        return data.get("result", {})

    async def close(self) -> None:
        """Cleanup."""
        await self._http.aclose()


async def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("Error: TELEGRAM_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    nous_url = os.environ.get("NOUS_API_URL", "http://localhost:8000")

    allowed_str = os.environ.get("NOUS_ALLOWED_USERS", "")
    allowed_users = None
    if allowed_str:
        allowed_users = {int(uid.strip()) for uid in allowed_str.split(",") if uid.strip()}

    bot = NousTelegramBot(bot_token, nous_url, allowed_users)
    try:
        await bot.start()
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
