"""Event handlers for Nous.

Handlers listen to bus events and react asynchronously.
Each handler registers itself on specific event types during __init__.
"""

from nous.config import Settings


def build_anthropic_headers(settings: Settings) -> dict[str, str]:
    """Build auth headers for Anthropic API calls.

    Shared by all handlers that make LLM calls (episode_summarizer,
    fact_extractor, sleep_handler).
    """
    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    api_key = getattr(settings, "anthropic_auth_token", None) or getattr(
        settings, "anthropic_api_key", None
    )
    if api_key and "sk-ant-oat" in api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
        headers["anthropic-dangerous-direct-browser-access"] = "true"
    else:
        headers["x-api-key"] = api_key or ""
    return headers
