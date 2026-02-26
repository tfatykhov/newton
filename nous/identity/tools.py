"""Identity tools — store_identity and complete_initiation.

Registered with ToolDispatcher during startup. Only available during
initiation via the "initiation" frame in FRAME_TOOLS.

Review fixes:
- P1-3: Plain async def + dispatcher.register() pattern (no @tool)
- P2-3: Validates section against VALID_SECTIONS
- Finding 6: Section enum matches IdentityManager.SECTIONS exactly
"""

from __future__ import annotations

import logging
from typing import Any

from nous.identity.manager import IdentityManager, VALID_SECTIONS, SECTIONS
from nous.identity.protocol import STORE_IDENTITY_SCHEMA, COMPLETE_INITIATION_SCHEMA
from nous.api.tools import ToolDispatcher

logger = logging.getLogger(__name__)


def _mcp_response(text: str) -> dict[str, Any]:
    """Build MCP-format response."""
    return {"content": [{"type": "text", "text": text}]}


def register_identity_tools(dispatcher: ToolDispatcher, identity_manager: IdentityManager) -> None:
    """Register identity tools with the dispatcher.

    These tools are gated by the "initiation" frame — only available
    during the initiation protocol.
    """

    async def store_identity(section: str, content: str) -> dict[str, Any]:
        """Store a piece of identity information."""
        # Review fix P2-3: validate section name
        if section not in SECTIONS:
            return _mcp_response(
                f"Invalid section '{section}'. Valid sections: {', '.join(SECTIONS)}"
            )

        try:
            await identity_manager.update_section(
                section, content, updated_by="initiation"
            )
            logger.info("Initiation: stored %s section (%d chars)", section, len(content))
            return _mcp_response(f"✅ Stored {section} identity.")
        except Exception as e:
            logger.error("Failed to store identity section %s: %s", section, e)
            return _mcp_response(f"Error storing {section}: {e}")

    async def complete_initiation() -> dict[str, Any]:
        """Mark initiation as complete."""
        try:
            # P2-4: Validate that at least character + user_profile exist
            identity = await identity_manager.get_identity()
            stored = set(identity.keys())
            required = {"character", "user_profile"}
            missing = required - stored
            if missing:
                return _mcp_response(
                    f"Cannot complete initiation — missing required sections: "
                    f"{', '.join(sorted(missing))}. "
                    f"Please store them first with store_identity."
                )
            await identity_manager.mark_initiated()
            identity_manager._invalidate_cache()
            logger.info("Initiation protocol completed for agent %s", identity_manager.agent_id)
            return _mcp_response(
                "✅ Initiation complete! I now have an identity and know who you are. "
                "Let's get to work!"
            )
        except Exception as e:
            logger.error("Failed to complete initiation: %s", e)
            return _mcp_response(f"Error completing initiation: {e}")

    dispatcher.register("store_identity", store_identity, STORE_IDENTITY_SCHEMA)
    dispatcher.register("complete_initiation", complete_initiation, COMPLETE_INITIATION_SCHEMA)
