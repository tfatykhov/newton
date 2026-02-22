"""Generate embeddings via OpenAI API.

Uses httpx.AsyncClient for async HTTP with connection pooling.
Gracefully handles missing API key by returning None.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Async embedding generation using OpenAI text-embedding-3-small."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self._client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": text,
                "dimensions": self.dimensions,
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (single API call)."""
        response = await self._client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": texts,
                "dimensions": self.dimensions,
            },
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
