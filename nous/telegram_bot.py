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
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Telegram Bot API base
TG_API = "https://api.telegram.org/bot{token}/{method}"

# Max Telegram message length
TG_MAX_LEN = 4096


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

        # Forward to Nous
        await self._chat(chat_id, text)

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
