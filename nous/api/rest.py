"""REST API for Nous agent.

Endpoints:
  POST /chat              - Send message, get response
  DELETE /chat/{session}  - End conversation
  GET  /status            - Agent status + memory stats + calibration
  GET  /decisions         - List recent decisions (Brain)
  GET  /decisions/{id}    - Get decision detail
  GET  /episodes          - List recent episodes (Heart)
  GET  /facts             - Search facts (Heart)
  GET  /censors           - Active censors (Heart)
  GET  /frames            - Available frames
  GET  /calibration       - Calibration report (Brain)
  GET  /health            - Health check (DB connectivity)
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from nous.api.runner import AgentRunner
from nous.brain import Brain
from nous.cognitive import CognitiveLayer
from nous.config import Settings
from nous.heart import Heart
from nous.storage.database import Database

logger = logging.getLogger(__name__)


def create_app(
    runner: AgentRunner,
    brain: Brain,
    heart: Heart,
    cognitive: CognitiveLayer,
    database: Database,
    settings: Settings,
    lifespan: Any | None = None,
) -> Starlette:
    """Create the Starlette ASGI app with all routes."""

    async def chat(request: Request) -> JSONResponse:
        """POST /chat - Send a message, get a response."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        message = body.get("message")
        if not message:
            return JSONResponse({"error": "Missing required field: message"}, status_code=400)

        session_id = body.get("session_id") or str(uuid4())

        try:
            response_text, turn_context = await runner.run_turn(session_id, message)
            return JSONResponse(
                {
                    "response": response_text,
                    "session_id": session_id,
                    "frame": turn_context.frame.frame_id,
                    "decision_id": turn_context.decision_id,
                }
            )
        except Exception as e:
            logger.error("Chat error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def end_chat(request: Request) -> JSONResponse:
        """DELETE /chat/{session_id} - End a conversation."""
        session_id = request.path_params["session_id"]
        try:
            await runner.end_conversation(session_id)
            return JSONResponse({"status": "ended", "session_id": session_id})
        except Exception as e:
            logger.error("End chat error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def status(request: Request) -> JSONResponse:
        """GET /status - Agent status overview."""
        try:
            calibration = await brain.get_calibration()

            # Raw SQL COUNT queries (F23)
            from sqlalchemy import text

            async with database.session() as session:
                counts = {}
                for table, key in [
                    ("brain.decisions", "total_decisions"),
                    ("heart.facts", "total_facts"),
                    ("heart.episodes", "total_episodes"),
                    ("heart.procedures", "total_procedures"),
                ]:
                    result = await session.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE agent_id = :agent_id"),
                        {"agent_id": settings.agent_id},
                    )
                    counts[key] = result.scalar() or 0

                # Active censors count
                result = await session.execute(
                    text("SELECT COUNT(*) FROM heart.censors WHERE agent_id = :agent_id AND active = true"),
                    {"agent_id": settings.agent_id},
                )
                counts["active_censors"] = result.scalar() or 0

            return JSONResponse(
                {
                    "agent_id": settings.agent_id,
                    "agent_name": settings.agent_name,
                    "model": settings.model,
                    "calibration": {
                        "brier_score": calibration.brier_score,
                        "accuracy": calibration.accuracy,
                        "total_decisions": calibration.total_decisions,
                        "reviewed_decisions": calibration.reviewed_decisions,
                    },
                    "memory": {
                        "active_conversations": len(runner._conversations),
                        "active_censors": counts["active_censors"],
                        "total_decisions": counts["total_decisions"],
                        "total_facts": counts["total_facts"],
                        "total_episodes": counts["total_episodes"],
                        "total_procedures": counts["total_procedures"],
                    },
                }
            )
        except Exception as e:
            logger.error("Status error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def list_decisions(request: Request) -> JSONResponse:
        """GET /decisions?limit=20&offset=0 - Recent decisions."""
        try:
            limit = int(request.query_params.get("limit", "20"))
            offset = int(request.query_params.get("offset", "0"))
        except ValueError:
            return JSONResponse({"error": "limit and offset must be integers"}, status_code=400)

        try:
            decisions, total = await brain.list_decisions(limit=limit, offset=offset)
            return JSONResponse(
                {
                    "decisions": [d.model_dump(mode="json") for d in decisions],
                    "total": total,
                }
            )
        except Exception as e:
            logger.error("List decisions error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def get_decision(request: Request) -> JSONResponse:
        """GET /decisions/{id} - Decision detail."""
        decision_id_str = request.path_params["id"]
        try:
            decision_id = UUID(decision_id_str)
        except ValueError:
            return JSONResponse({"error": "Invalid decision ID"}, status_code=400)

        try:
            detail = await brain.get(decision_id)
            if detail is None:
                return JSONResponse({"error": "Decision not found"}, status_code=404)
            return JSONResponse(detail.model_dump(mode="json"))
        except Exception as e:
            logger.error("Get decision error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def list_episodes(request: Request) -> JSONResponse:
        """GET /episodes?limit=20 - Recent episodes."""
        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)

        try:
            episodes = await heart.list_episodes(limit=limit)
            return JSONResponse(
                {
                    "episodes": [e.model_dump(mode="json") for e in episodes],
                }
            )
        except Exception as e:
            logger.error("List episodes error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def search_facts(request: Request) -> JSONResponse:
        """GET /facts?q=query&limit=20 - Search facts."""
        q = request.query_params.get("q")
        if not q:
            return JSONResponse({"error": "Missing required query parameter: q"}, status_code=400)

        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)

        try:
            facts = await heart.search_facts(q, limit=limit)
            return JSONResponse(
                {
                    "facts": [f.model_dump(mode="json") for f in facts],
                }
            )
        except Exception as e:
            logger.error("Search facts error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def list_censors(request: Request) -> JSONResponse:
        """GET /censors - Active censors."""
        try:
            censors = await heart.list_censors()
            return JSONResponse(
                {
                    "censors": [c.model_dump(mode="json") for c in censors],
                }
            )
        except Exception as e:
            logger.error("List censors error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def list_frames(request: Request) -> JSONResponse:
        """GET /frames - Available cognitive frames."""
        try:
            frames = await cognitive.list_frames(settings.agent_id)
            return JSONResponse(
                {
                    "frames": [f.model_dump(mode="json") for f in frames],
                }
            )
        except Exception as e:
            logger.error("List frames error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def calibration(request: Request) -> JSONResponse:
        """GET /calibration - Full calibration report."""
        try:
            report = await brain.get_calibration()
            return JSONResponse(report.model_dump(mode="json"))
        except Exception as e:
            logger.error("Calibration error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    async def health(request: Request) -> JSONResponse:
        """GET /health - Health check."""
        try:
            from sqlalchemy import text

            async with database.session() as session:
                await session.execute(text("SELECT 1"))
            return JSONResponse({"status": "healthy"})
        except Exception as e:
            return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

    routes = [
        Route("/chat", chat, methods=["POST"]),
        Route("/chat/{session_id}", end_chat, methods=["DELETE"]),
        Route("/status", status),
        Route("/decisions", list_decisions),
        Route("/decisions/{id}", get_decision),
        Route("/episodes", list_episodes),
        Route("/facts", search_facts),
        Route("/censors", list_censors),
        Route("/frames", list_frames),
        Route("/calibration", calibration),
        Route("/health", health),
    ]

    kwargs: dict[str, Any] = {"routes": routes}
    if lifespan is not None:
        kwargs["lifespan"] = lifespan
    return Starlette(**kwargs)
