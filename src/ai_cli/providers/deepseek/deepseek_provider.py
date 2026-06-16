"""DeepSeek provider for ai_cli supporting chat and embeddings."""
from __future__ import annotations

import os

from openai import OpenAI

from ..base import BaseProvider, ProviderConfig
from ..contracts import ChatProvider, EmbeddingProvider
from ..registry import register_provider


@register_provider("deepseek")
class DeepSeekProvider(BaseProvider, ChatProvider, EmbeddingProvider):
    """DeepSeek AI provider with chat and embedding support.

    Uses the OpenAI-compatible DeepSeek API.
    """

    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_EMBED = "text-embedding-3-small"

    def __init__(self, config: ProviderConfig | None = None, **kwargs) -> None:
        """Initialise DeepSeekProvider.

        Args:
            config: Optional provider configuration.
            **kwargs: Passed to ProviderConfig if *config* is not given.
        """
        config = config or ProviderConfig(**kwargs)
        super().__init__(config)
        self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY missing")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )
        self.embed_model = config.embedding_model or self.DEFAULT_EMBED

    # ---------------- CHAT ----------------

    def chat(self, prompt: str, **kwargs) -> str:
        """Send *prompt* to DeepSeek and return the response text.

        Args:
            prompt: User message to send.
            **kwargs: Extra parameters forwarded to the completions endpoint.

        Returns:
            Stripped response string from the model.
        """
        resp = self.client.chat.completions.create(
            model=self.model or self.DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content.strip()

    # -------------- EMBED -----------------

    def embed(self, texts: list[str], **_kwargs) -> list[list[float]]:
        """Return embeddings for *texts* using the DeepSeek embedding model.

        Args:
            texts: Strings to embed.
            **_kwargs: Ignored; present for interface compatibility.

        Returns:
            List of float embedding vectors, one per input string.
        """
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]