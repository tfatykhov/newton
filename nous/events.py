"""In-process async event bus for Nous.

Events are dispatched to registered handlers asynchronously.
Handlers run concurrently but errors are isolated — one broken
handler never crashes the bus or blocks other handlers.

Also persists events to DB (existing behavior) for audit trail.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Handler type: async function taking an Event
EventHandler = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    """A typed event flowing through the bus."""

    type: str
    agent_id: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class EventBus:
    """In-process async event bus with error isolation.

    Events are queued and processed by a background asyncio task.
    Handlers registered via on() are called concurrently for each event.
    Handler errors are logged but never propagate.

    The bus also delegates to a DB persister (the existing emit_event
    pattern) so all events remain in the audit table.
    """

    def __init__(self, max_queue: int = 1000):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._running = False
        self._db_persister: EventHandler | None = None

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for an event type. Can register multiple."""
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler for '%s': %s", event_type, handler.__qualname__)

    def set_db_persister(self, persister: EventHandler) -> None:
        """Set the DB persistence handler (existing Brain.emit_event pattern)."""
        self._db_persister = persister

    async def emit(self, event: Event) -> None:
        """Emit an event. Non-blocking — queued for async processing.

        If queue is full, logs warning and drops event (never blocks caller).
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event bus queue full, dropping event: %s", event.type)

    async def start(self) -> None:
        """Start the background processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop(), name="event-bus")
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop the bus. Drains remaining events before stopping.

        P1-2 fix: Cancel task first, await it, THEN drain remaining events.
        """
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            # THEN drain remaining events
            while not self._queue.empty():
                try:
                    event = self._queue.get_nowait()
                    await self._dispatch(event)
                except asyncio.QueueEmpty:
                    break
        logger.info("Event bus stopped")

    async def _process_loop(self) -> None:
        """Main processing loop — runs as background task."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in event bus loop")

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all registered handlers + DB persister."""
        # DB persistence (fire-and-forget, errors logged)
        if self._db_persister:
            try:
                await self._db_persister(event)
            except Exception:
                logger.warning("DB persist failed for event %s", event.type)

        # Handlers — run concurrently, errors isolated
        handlers = self._handlers.get(event.type, [])
        if not handlers:
            return

        tasks = [self._safe_handle(h, event) for h in handlers]
        await asyncio.gather(*tasks)

    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        """Run handler with error isolation. Never propagates (except CancelledError).

        P0-13 fix: catch BaseException, re-raise CancelledError.
        """
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise  # Propagate cancellation
        except BaseException:
            logger.exception(
                "Handler %s failed for event %s",
                handler.__qualname__,
                event.type,
            )

    @property
    def pending(self) -> int:
        """Number of events waiting in queue."""
        return self._queue.qsize()
