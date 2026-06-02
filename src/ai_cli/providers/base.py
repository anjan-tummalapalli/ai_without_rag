"""Base classes for AI provider integrations with Advanced RAG support.

This module defines an abstract AIProvider used by provider-specific
integrations. It centralizes prompt validation, retry handling,
response coercion, response validation, hallucination detection, and
model quality metrics so individual providers only implement transport
and API specifics.

Advanced RAG features:
- Chunking utilities to split large documents into overlapping chunks.
- Embedding client abstraction hooks (any object providing `embed(list[str]) -> list[list[float]]`).
- Vector DB abstraction hooks (expected methods: `upsert(ids, embeddings, metadatas)`,
    `query_embedding(embedding, top_k) -> list[dict{id, text, score}]`, optionally `get(id)`).
- High-level helpers to index documents, retrieve top-k relevant chunks, and compose
    augmented prompts so providers can perform Retrieval-Augmented Generation (RAG).
- Simple default behavior if embedding/vector clients are not provided (falls back to pure LLM send).

End result: calling send(prompt) returns a validated, coerced string response (or raises
ProviderRequestError on failure) while updating metrics and logging relevant events.
Providers can call send_rag(prompt, top_k=...) to perform retrieval and augment the prompt
with context before sending to the underlying model.
"""

from __future__ import annotations
import time, uuid, logging, json
from dataclasses import dataclass
from typing import Optional, Iterable, List, Any, Tuple

from ai_cli.core.exceptions import (
        ProviderConfigurationError,
        ProviderRequestError,
        PromptValidationError,
)
from ai_cli.core.resilience import RetryEngine
from ai_cli.utils.validation import ResponseValidator, HallucinationDetector
from ai_cli.telemetry.monitoring import ModelQualityMetrics

logger = logging.getLogger("ai_gateway")

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_PROMPT_LENGTH = 10_000


@dataclass(frozen=True)
class ProviderMetadata:
        """Metadata describing a provider and its capabilities."""

        name: str
        default_model: str
        supported_models: list[str]
        supports_streaming: bool
        supports_tools: bool
        supports_vision: bool
        max_context: int
        cost_per_1k_tokens: float
        avg_latency_ms: int
        supports_rag: bool = False  # whether the provider has first-class RAG support


class AIProvider:
        """Abstract base for provider integrations with optional RAG helpers.

        To enable RAG you can provide:
        - embedding_client: any object with embed(list[str]) -> list[list[float]]
        - vector_db: any object with upsert(ids, embeddings, metadatas) and query_embedding(embedding, top_k)
        The concrete implementations are intentionally generic to allow different embedding
        and vectorstore libraries to be plugged in.
        """

        def __init__(
                self,
                provider_name: str,
                model: str | None = None,
                timeout: int = DEFAULT_TIMEOUT_SECONDS,
                provider_meta: ProviderMetadata | None = None,
                *,
                embedding_client: Optional[Any] = None,
                vector_db: Optional[Any] = None,
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
        ) -> None:
                """Initialize AIProvider base."""
                if provider_meta is None:
                        raise ProviderConfigurationError("Provider metadata is required")

                self.provider_name = provider_name
                self.timeout = timeout
                self.model = model or provider_meta.default_model
                self.trace_id = str(uuid.uuid4())
                self.retry_engine = RetryEngine()
                self.response_validator = ResponseValidator()
                self.hallucination_detector = HallucinationDetector()
                self.metrics = ModelQualityMetrics(
                        provider=provider_name, model=self.model
                )
                self._provider_meta = provider_meta

                # RAG components (optional)
                self.embedding_client = embedding_client
                self.vector_db = vector_db
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

        def validate_prompt(self, prompt: str) -> str:
                """Validate, sanitize, and correct a prompt string."""
                if not isinstance(prompt, str):
                        raise PromptValidationError("prompt must be string")
                if "\x00" in prompt:
                        raise PromptValidationError("prompt contains NUL byte")

                from ai_cli.core.prompt_corrector import prompt_corrector
                corrected = prompt_corrector.correct(prompt)

                if not corrected:
                        raise PromptValidationError("prompt is empty")
                if len(corrected) > DEFAULT_MAX_PROMPT_LENGTH:
                        raise PromptValidationError("prompt exceeds maximum length")
                return corrected

        def _send_impl(self, _prompt: str) -> str:
                """Provider-specific implementation must override."""
                raise NotImplementedError(
                        f"{self.__class__.__name__} must implement _send_impl()"
                )

        def _coerce_response_to_str(self, response) -> str:
                """Handle non-string responses from providers and coerce to str."""
                if response is None:
                        raise ProviderRequestError(
                                f"{self.provider_name} returned empty response (None)"
                        )
                if isinstance(response, str):
                        return response
                if isinstance(response, bytes):
                        try:
                                return response.decode("utf-8")
                        except Exception:
                                return response.decode("latin-1", errors="ignore")
                if isinstance(response, (dict, list, tuple, int, float, bool)):
                        try:
                                return json.dumps(response, ensure_ascii=False)
                        except Exception as exc:
                                logger.warning(
                                        "response_serialization_failed provider=%s error=%s",
                                        self.provider_name,
                                        exc,
                                )
                                # Fallback to str()
                                return str(response)
                # Best-effort fallback for arbitrary objects
                try:
                        return str(response)
                except Exception as exc:
                        logger.exception(
                                "response_coercion_failed provider=%s trace_id=%s error=%s",
                                self.provider_name,
                                self.trace_id,
                                exc,
                        )
                        raise ProviderRequestError(
                                f"{self.provider_name} returned an unsupported response type: {type(response)}"
                        ) from exc

        def send(self, prompt: str) -> str:
                """High-level send that validates prompt, runs retries, and checks."""
                validated_prompt = self.validate_prompt(prompt)
                self.metrics.requests += 1
                start_time = time.monotonic()

                logger.info(
                        "provider_request provider=%s model=%s trace_id=%s",
                        self.provider_name,
                        self.model,
                        self.trace_id,
                )

                try:
                        raw_response = self.retry_engine.execute(
                                lambda: self._send_impl(validated_prompt)
                        )

                        # Coerce non-string responses (e.g., dicts, bytes, numbers) into a string
                        response = self._coerce_response_to_str(raw_response)

                        duration = time.monotonic() - start_time
                        self.metrics.total_latency_seconds += duration
                        self.response_validator.validate(response)

                        hallucination = self.hallucination_detector.evaluate(response)
                        if not hallucination.passed:
                                self.metrics.hallucination_failures += 1
                                logger.warning(
                                        "hallucination_detected provider=%s score=%s reasons=%s",
                                        self.provider_name,
                                        hallucination.score,
                                        hallucination.reasons,
                                )
                        return response.strip()
                except Exception as exc:
                        self.metrics.failures += 1
                        logger.exception(
                                "provider_error provider=%s trace_id=%s error=%s",
                                self.provider_name,
                                self.trace_id,
                                exc,
                        )
                        raise ProviderRequestError(
                                f"{self.provider_name} request failed: {exc}"
                        ) from exc

        # -----------------------
        # RAG helper utilities
        # -----------------------
        def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
                """Split `text` into overlapping chunks.

                Uses simple sliding-window chunking. Returns list of chunk strings.
                """
                if chunk_size is None:
                        chunk_size = self.chunk_size
                if overlap is None:
                        overlap = self.chunk_overlap
                if chunk_size <= 0:
                        raise ValueError("chunk_size must be > 0")
                if overlap < 0 or overlap >= chunk_size:
                        overlap = max(0, chunk_size // 10)

                tokens = text.split()
                if not tokens:
                        return []

                chunks: List[str] = []
                start = 0
                while start < len(tokens):
                        end = min(len(tokens), start + chunk_size)
                        chunk = " ".join(tokens[start:end])
                        chunks.append(chunk)
                        if end == len(tokens):
                                break
                        start = end - overlap
                        if start < 0:
                                start = 0
                return chunks

        def index_documents(self, docs: Iterable[str], ids_prefix: str = None) -> List[str]:
                """Create embeddings for docs and upsert into vector_db.

                Returns list of upserted ids. Raises ProviderRequestError if embedding_client
                or vector_db are not configured.
                """
                if self.embedding_client is None or self.vector_db is None:
                        raise ProviderRequestError("embedding_client and vector_db are required for indexing")

                docs = list(docs)
                # Chunk docs before embedding for better retrieval granularity
                all_chunks: List[str] = []
                for i, d in enumerate(docs):
                        chunks = self.chunk_text(d)
                        # annotate chunk with source doc index for traceability
                        annotated = [f"doc:{i} chunk:{j} {c}" for j, c in enumerate(chunks)]
                        all_chunks.extend(annotated)

                if not all_chunks:
                        return []

                try:
                        embeddings = self.embedding_client.embed(all_chunks)
                except Exception as exc:
                        logger.exception("embedding_generation_failed provider=%s error=%s", self.provider_name, exc)
                        raise ProviderRequestError("failed to generate embeddings") from exc

                ids = []
                for idx in range(len(all_chunks)):
                        id_ = f"{ids_prefix or self.provider_name}-{uuid.uuid4()}"
                        ids.append(id_)

                try:
                        # vector_db expected to accept upsert(ids, embeddings, metadatas)
                        metadatas = [{"text": txt} for txt in all_chunks]
                        self.vector_db.upsert(ids, embeddings, metadatas)
                except Exception as exc:
                        logger.exception("vectordb_upsert_failed provider=%s error=%s", self.provider_name, exc)
                        raise ProviderRequestError("failed to upsert embeddings to vector DB") from exc

                return ids

        def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
                """Retrieve top_k relevant chunks for `query`.

                Returns list of tuples (text, score). If embedding/vector DB not configured
                falls back to empty list.
                """
                if self.embedding_client is None or self.vector_db is None:
                        logger.debug("retrieve: embedding/vector DB not configured, returning empty context")
                        return []

                try:
                        q_emb = self.embedding_client.embed([query])[0]
                except Exception as exc:
                        logger.exception("embedding_generation_failed provider=%s error=%s", self.provider_name, exc)
                        return []

                try:
                        results = self.vector_db.query_embedding(q_emb, top_k=top_k)
                except Exception as exc:
                        logger.exception("vectordb_query_failed provider=%s error=%s", self.provider_name, exc)
                        return []

                # Expect results as list of dicts with keys: 'text' and 'score'
                out = []
                for r in results:
                        text = r.get("text") if isinstance(r, dict) else getattr(r, "text", None)
                        score = r.get("score", 0.0) if isinstance(r, dict) else getattr(r, "score", 0.0)
                        if text:
                                out.append((text, float(score)))
                return out

        def build_augmented_prompt(self, prompt: str, context_chunks: Iterable[str]) -> str:
                """Compose augmented prompt with retrieved context.

                Simple default: prepend "Context:\n<chunks>\n---\n" before the prompt.
                Providers can override to implement more advanced prompt templates.
                """
                chunks = list(context_chunks)
                if not chunks:
                        return prompt
                context_text = "\n\n".join(chunks)
                augmented = f"Context:\n{context_text}\n\n---\n\n{prompt}"
                return augmented

        def send_rag(self, prompt: str, top_k: int = 5) -> str:
                """High-level send that performs retrieval augmentation then calls send().

                If RAG components aren't configured this method falls back to send().
                """
                validated_prompt = self.validate_prompt(prompt)
                if self.embedding_client is None or self.vector_db is None:
                        logger.info("send_rag: RAG components missing, falling back to plain send")
                        return self.send(validated_prompt)

                # Retrieve relevant chunks
                retrieved = self.retrieve(validated_prompt, top_k=top_k)
                contexts = [t for t, _score in retrieved]
                augmented = self.build_augmented_prompt(validated_prompt, contexts)

                # Provide the composed augmented prompt to the regular send path (keeps metrics, retries, validation)
                return self.send(augmented)


class EchoProvider(AIProvider):
        """Local echo provider used for testing and defaults.

        EchoProvider supports RAG flows in a minimal way: if a vector_db/embedding_client
        are attached to it, send_rag will include retrieved chunks in the composed prompt;
        otherwise it behaves as a simple echo.
        """

        def __init__(self, model: str | None = None, *, embedding_client: Optional[Any] = None, vector_db: Optional[Any] = None) -> None:
                """Initialize EchoProvider."""
                meta = ProviderMetadata(
                        name="Local Echo",
                        default_model="echo",
                        supported_models=["echo"],
                        supports_streaming=False,
                        supports_tools=False,
                        supports_vision=False,
                        max_context=1_000_000,
                        cost_per_1k_tokens=0.0,
                        avg_latency_ms=1,
                        supports_rag=True,
                )
                super().__init__(provider_name="echo", model=model, provider_meta=meta, embedding_client=embedding_client, vector_db=vector_db)

        def _send_impl(self, prompt: str) -> str:
                """Return echoed prompt. If prompt includes context markers, preserves them."""
                return f"(echo) {prompt}"
