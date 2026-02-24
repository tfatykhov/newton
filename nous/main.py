"""Nous agent entry point.

Initializes all components and starts the server:
  Settings -> Database -> Brain -> Heart -> CognitiveLayer -> Runner -> App -> Uvicorn

Uses Starlette lifespan to manage component lifecycle on the same
event loop as uvicorn (F2/F3 fix from 3-agent review).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

from nous.api.builtin_tools import register_builtin_tools
from nous.api.runner import AgentRunner
from nous.api.tools import ToolDispatcher, register_nous_tools
from nous.api.web_tools import register_web_tools
from nous.brain import Brain
from nous.brain.embeddings import EmbeddingProvider
from nous.cognitive import CognitiveLayer
from nous.config import Settings
from nous.events import Event, EventBus
from nous.heart import Heart
from nous.storage.database import Database

logger = logging.getLogger(__name__)


async def create_components(settings: Settings) -> dict:
    """Initialize all components in dependency order.

    Returns dict with all components for lifespan storage.

    1. Database - connection pool
    2. EmbeddingProvider - optional (None if no API key)
    3. Brain - decision intelligence
    4. Heart - memory system (owns_embeddings=False per F4)
    5. CognitiveLayer - orchestrator
    6. AgentRunner - LLM integration
    """
    database = Database(settings)
    await database.connect()  # F1: connect() not initialize()

    embedding_provider = None
    if settings.openai_api_key:
        embedding_provider = EmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    brain = Brain(database, settings, embedding_provider)
    heart = Heart(database, settings, embedding_provider, owns_embeddings=False)  # F4

    # 006: Create EventBus (only if enabled)
    bus = None
    handler_http = None
    session_monitor = None
    if settings.event_bus_enabled:
        bus = EventBus()

        # DB persistence adapter (P0-1 fix: correct signature — no agent_id/session_id kwargs)
        async def persist_to_db(event: Event) -> None:
            data = {**event.data}
            if event.session_id:
                data["session_id"] = event.session_id
            await brain.emit_event(event.type, data)

        bus.set_db_persister(persist_to_db)

    # P0-2/P0-3 fix: preserve identity_prompt, pass bus as keyword arg
    cognitive = CognitiveLayer(brain, heart, settings, settings.identity_prompt, bus=bus)

    # 006: Register handlers on bus (after cognitive exists for monitor)
    if bus is not None:
        handler_http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=60, write=10, pool=10),
        )

        try:
            from nous.handlers.episode_summarizer import EpisodeSummarizer

            if settings.episode_summary_enabled:
                EpisodeSummarizer(heart, settings, bus, handler_http)
        except ImportError:
            logger.debug("EpisodeSummarizer not available yet")

        try:
            from nous.handlers.fact_extractor import FactExtractor

            if settings.fact_extraction_enabled:
                FactExtractor(heart, settings, bus, handler_http)
        except ImportError:
            logger.debug("FactExtractor not available yet")

        try:
            from nous.handlers.session_monitor import SessionTimeoutMonitor

            session_monitor = SessionTimeoutMonitor(bus, settings, cognitive=cognitive)
        except ImportError:
            logger.debug("SessionTimeoutMonitor not available yet")

        try:
            from nous.handlers.sleep_handler import SleepHandler

            if settings.sleep_enabled:
                SleepHandler(brain, heart, settings, bus, handler_http)
        except ImportError:
            logger.debug("SleepHandler not available yet")

        # Start bus + monitor
        await bus.start()
        if session_monitor:
            await session_monitor.start()

    # Create tool dispatcher and register all tools
    dispatcher = ToolDispatcher()
    register_nous_tools(dispatcher, brain, heart)
    register_builtin_tools(dispatcher, settings)

    # Web tools httpx client (separate from runner — no API auth headers)
    web_http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=30, write=10, pool=10),
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
    )
    register_web_tools(dispatcher, settings, web_http)

    runner = AgentRunner(cognitive, brain, heart, settings)
    runner.set_dispatcher(dispatcher)
    await runner.start()

    return {
        "database": database,
        "brain": brain,
        "heart": heart,
        "cognitive": cognitive,
        "runner": runner,
        "dispatcher": dispatcher,
        "embedding_provider": embedding_provider,
        "web_http": web_http,
        "bus": bus,
        "session_monitor": session_monitor,
        "handler_http": handler_http,
    }


async def shutdown_components(components: dict) -> None:
    """Graceful shutdown in reverse order."""
    logger.info("Shutting down Nous...")

    # 006: Stop session monitor and event bus first
    session_monitor = components.get("session_monitor")
    if session_monitor:
        await session_monitor.stop()

    bus = components.get("bus")
    if bus:
        await bus.stop()

    handler_http = components.get("handler_http")
    if handler_http:
        await handler_http.aclose()

    web_http = components.get("web_http")
    if web_http:
        await web_http.aclose()

    runner = components.get("runner")
    if runner:
        await runner.close()

    heart = components.get("heart")
    if heart:
        await heart.close()

    brain = components.get("brain")
    if brain:
        await brain.close()

    database = components.get("database")
    if database:
        await database.disconnect()  # F1: disconnect() not close()

    logger.info("Nous shutdown complete.")


def build_app(settings: Settings) -> Starlette:
    """Build the combined Starlette app with REST + MCP.

    Uses Starlette lifespan for component lifecycle management (F2/F3).
    """
    # Closure to share components between lifespan and app
    components: dict = {}

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        # Startup
        nonlocal components
        components.update(await create_components(settings))

        # Store on app.state for access in tests
        app.state.components = components

        logger.info("Nous started: %s (%s)", settings.agent_name, settings.agent_id)
        logger.info(
            "API: max_turns=%d, workspace=%s",
            settings.max_turns,
            settings.workspace_dir,
        )
        yield

        # Shutdown (reverse order)
        # MCP session manager cleanup (F25)
        mcp_manager = getattr(app.state, "mcp_manager", None)
        if mcp_manager:
            try:
                await mcp_manager.close()
            except Exception:
                logger.warning("MCP session manager cleanup failed")

        await shutdown_components(components)

    # Import here to avoid circular imports at module level
    from nous.api.rest import create_app

    app = create_app(
        runner=_lazy_component(components, "runner"),
        brain=_lazy_component(components, "brain"),
        heart=_lazy_component(components, "heart"),
        cognitive=_lazy_component(components, "cognitive"),
        database=_lazy_component(components, "database"),
        settings=settings,
        lifespan=lifespan,
    )

    if settings.mcp_enabled:
        try:
            from nous.api.mcp import create_mcp_server

            # MCP server needs real components, which are only available after lifespan starts.
            # We mount a lazy ASGI app that creates the MCP server on first request.
            _mcp_manager = None

            async def mcp_asgi(scope, receive, send):
                nonlocal _mcp_manager
                if _mcp_manager is None:
                    _mcp_manager = create_mcp_server(
                        runner=components["runner"],
                        brain=components["brain"],
                        heart=components["heart"],
                        settings=settings,
                    )
                    app.state.mcp_manager = _mcp_manager
                await _mcp_manager.handle_request(scope, receive, send)

            app.routes.append(Mount("/mcp", app=mcp_asgi))
            logger.info("MCP server mounted at /mcp")
        except ImportError:
            logger.warning("MCP dependencies not installed, skipping MCP server")

    return app


class _LazyProxy:
    """Proxy that defers attribute access to a dict-backed component.

    Allows create_app() to receive component references before lifespan
    has initialized them. All attribute access is forwarded to the actual
    component once it's available.
    """

    def __init__(self, components: dict, key: str) -> None:
        object.__setattr__(self, "_components", components)
        object.__setattr__(self, "_key", key)

    def _resolve(self):
        components = object.__getattribute__(self, "_components")
        key = object.__getattribute__(self, "_key")
        obj = components.get(key)
        if obj is None:
            raise RuntimeError(f"Component '{key}' not yet initialized — lifespan hasn't started")
        return obj

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __len__(self):
        return len(self._resolve())


def _lazy_component(components: dict, key: str) -> _LazyProxy:
    """Create a lazy proxy for a component that will be initialized in lifespan."""
    return _LazyProxy(components, key)


def main() -> None:
    """Entry point — parse settings, build app, run server."""
    settings = Settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Starting Nous agent: %s (%s)", settings.agent_name, settings.agent_id)
    logger.info("Model: %s", settings.model)
    logger.info("Database: %s:%s/%s", settings.db_host, settings.db_port, settings.db_name)
    logger.info("MCP: %s", "enabled" if settings.mcp_enabled else "disabled")

    # F15: Warn if no Anthropic credentials set
    if not settings.anthropic_api_key and not settings.anthropic_auth_token:
        logger.warning(
            "Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set — "
            "/chat endpoints will fail"
        )

    if not settings.brave_search_api_key:
        logger.warning("BRAVE_SEARCH_API_KEY not set — web_search will be unavailable")

    app = build_app(settings)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
