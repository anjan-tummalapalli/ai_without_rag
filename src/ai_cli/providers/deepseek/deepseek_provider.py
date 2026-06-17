from __future__ import annotations

import os
from typing import Any

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import BaseProvider, ProviderMetadata
from ai_cli.providers.registry import register_chat_provider, register_provider


# Simple in-memory "DeepSeek" provider used as a stable fallback.
# Implements minimal document upsert/retrieve + chat send/ask.
class DeepSeekProvider(BaseProvider):
    metadata = ProviderMetadata(name="deepseek")

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        # in-memory store: list of (id, text, metadata)
        self._docs: list[dict[str, Any]] = []

    def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        # Explicitly not implemented; keep attribute so contract tests pass.
        raise NotImplementedError("embeddings not supported by DeepSeekProvider")

    def send(self, prompt: str, **kwargs: Any) -> str:
        # Deterministic echo-like behavior for chat interface.
        if not isinstance(prompt, str):
            raise ProviderRequestError("prompt must be a string")
        return f"(deepseek) {prompt}"

    def ask(self, prompt: str, **kwargs: Any) -> str:
        return self.send(prompt, **kwargs)

    def upsert_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("texts and metadatas length mismatch")
        for txt, md in zip(texts, metadatas, strict=False):
            self._docs.append({"text": txt, "metadata": md})

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query:
            return []
        # simple substring scoring; higher score for earlier match
        scored: list[tuple[int, dict[str, Any]]] = []
        for _, doc in enumerate(self._docs):
            txt = doc.get("text", "") or ""
            pos = txt.lower().find(query.lower())
            if pos >= 0:
                score = 1_000_000 - pos  # earlier match => higher score
                scored.append((score, {"score": float(score), **doc}))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in scored[:top_k]]


register_provider("deepseek", DeepSeekProvider)
register_chat_provider("deepseek", DeepSeekProvider)