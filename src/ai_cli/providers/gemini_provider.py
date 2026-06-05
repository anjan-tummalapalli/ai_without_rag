"""
Optimized Gemini provider implementation for ai_cli with Advanced RAG
support.

Optimizations:
- InMemoryVectorDB precomputes norms to avoid repeated sqrt during queries
        and uses heapq.nlargest for top_k.
- chunk_text uses a safe stepping strategy and avoids unnecessary checks.
- _create_embeddings normalizes access to SDK response and supports single/
        multiple inputs robustly.
- Minor error handling and small performance/clarity improvements across
        methods.
"""

from __future__ import annotations

import os, math, heapq
from typing import Optional, Sequence, List, Dict, Any

from ai_cli.core.exceptions import ProviderRequestError

try:
                import google.generativeai as genai  # type: ignore
except Exception:
                # Minimal shim so static analysis and runtime usage produce clear
                # errors when the SDK is missing.
                class _GenaiShim:
                                def configure(self, *args, **kwargs):
                                        raise ProviderRequestError(
                                                "google-generativeai SDK is not installed; please install it "
                                                "(pip install google-generativeai)"
                                        )

                                class GenerativeModel:
                                                def __init__(self, *args, **kwargs):
                                                        raise ProviderRequestError(
                                                                "google-generativeai SDK is not installed; please install "
                                                                "it (pip install google-generativeai)"
                                                        )

                                class embeddings:
                                                @staticmethod
                                                def create(*args, **kwargs):
                                                                raise ProviderRequestError(
                                                                        "google-generativeai SDK is not installed; cannot create embeddings"
                                                                )

                genai = _GenaiShim()

from ai_cli.providers.base import AIProvider


class InMemoryVectorDB:
                """
                Minimal in-memory vector DB for storing embeddings.

                Stored items: list of dicts with keys:
                                - id: unique id (str)
                                - vector: List[float]
                                - norm: precomputed L2 norm (float)
                                - metadata: dict
                                - text: original chunk text
                """

                def __init__(self) -> None:
                                self._items: List[Dict[str, Any]] = []

                def upsert(self, items: Sequence[Dict[str, Any]]) -> None:
                                # Replace items with same id or append otherwise; precompute norms.
                                ids = {it["id"] for it in items}
                                self._items = [it for it in self._items if it["id"] not in ids]
                                for it in items:
                                                vec = it.get("vector", [])
                                                norm = math.sqrt(sum(x * x for x in vec)) if vec else 0.0
                                                stored = {
                                                                "id": it["id"],
                                                                "vector": list(vec),
                                                                "norm": norm,
                                                                "metadata": it.get("metadata", {}),
                                                                "text": it.get("text", ""),
                                                }
                                                self._items.append(stored)

                @staticmethod
                def _cosine_similarity_with_norms(
                                a: Sequence[float],
                                a_norm: float,
                                b: Sequence[float],
                                b_norm: float,
                ) -> float:
                                if a_norm == 0.0 or b_norm == 0.0:
                                                return 0.0
                                dot = 0.0
                                # zip will stop at shortest; vectors should match length but be
                                # defensive
                                for x, y in zip(a, b):
                                                dot += x * y
                                return dot / (a_norm * b_norm)

                def query(self, vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
                                if not self._items:
                                                return []
                                vec = list(vector)
                                vec_norm = math.sqrt(sum(x * x for x in vec)) if vec else 0.0

                                scored = []
                                for it in self._items:
                                                score = self._cosine_similarity_with_norms(
                                                                vec, vec_norm, it["vector"], it["norm"]
                                                )
                                                scored.append(
                                                                {
                                                                                "id": it["id"],
                                                                                "score": score,
                                                                                "metadata": it.get("metadata", {}),
                                                                                "text": it.get("text", ""),
                                                                }
                                                )

                                # fast top_k selection
                                if top_k >= len(scored):
                                                scored.sort(key=lambda x: x["score"], reverse=True)
                                                return scored
                                return heapq.nlargest(top_k, scored, key=lambda x: x["score"])


class GeminiProvider(AIProvider):
                """
                AI provider implementation for Google Gemini models with optional RAG.
                """

                def __init__(
                                self,
                                model: Optional[str] = None,
                                api_key: Optional[str] = None,
                                embedding_model: Optional[str] = None,
                                vector_db_client: Optional[Any] = None,
                                chunk_size: int = 500,
                                chunk_overlap: int = 50,
                                *args,
                                **kwargs,
                ) -> None:
                                super().__init__(
                                                model=model or "gemini-1.5-flash", api_key=api_key, *args, **kwargs
                                )
                                self.api_key = api_key or os.getenv("GEMINI_API_KEY")
                                if not self.api_key:
                                                raise ProviderRequestError("GEMINI_API_KEY environment variable is not set")

                                # Configure Gemini SDK and client
                                genai.configure(api_key=self.api_key)
                                self.client = genai.GenerativeModel(self.model)

                                # RAG-related
                                self.embedding_model = embedding_model
                                self.vector_db = vector_db_client or InMemoryVectorDB()
                                if chunk_size <= 0:
                                                raise ValueError("chunk_size must be positive")
                                if chunk_overlap < 0:
                                                raise ValueError("chunk_overlap must be non-negative")
                                self.chunk_size = chunk_size
                                self.chunk_overlap = chunk_overlap

                # -----------------------
                # Core send & health
                # -----------------------
                def _send_impl(self, prompt: str) -> str:
                                try:
                                                response = self.client.generate_content(prompt)
                                                if not response:
                                                        raise ProviderRequestError("Gemini returned empty response")
                                                # tolerant access to text
                                                text = getattr(response, "text", None) or (
                                                        response.get("text") if isinstance(response, dict) else None
                                                )
                                                if not text:
                                                        raise ProviderRequestError("Gemini response missing text field")
                                                return text.strip()
                                except Exception as exc:
                                                raise ProviderRequestError(f"Gemini request failed: {exc}") from exc

                def health_check(self) -> bool:
                                try:
                                                response = self.client.generate_content("ping")
                                                text = getattr(response, "text", None) or (
                                                                response.get("text") if isinstance(response, dict) else None
                                                )
                                                return bool(text)
                                except Exception:
                                                return False

                @property
                def provider_name(self) -> str:
                                return "gemini"

                # -----------------------
                # Chunking utilities
                # -----------------------
                def chunk_text(
                                self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
                ) -> List[str]:
                                if chunk_size is None:
                                                chunk_size = self.chunk_size
                                if overlap is None:
                                                overlap = self.chunk_overlap

                                # ensure progress on each loop
                                step = max(chunk_size - overlap, 1)
                                text_len = len(text)
                                if text_len == 0:
                                                return []

                                chunks: List[str] = []
                                for start in range(0, text_len, step):
                                                end = start + chunk_size
                                                chunks.append(text[start:end])
                                                if end >= text_len:
                                                                break
                                return chunks

                # -----------------------
                # Embeddings utilities
                # -----------------------
                def _create_embeddings(self, inputs: Sequence[str]) -> List[List[float]]:
                                if not inputs:
                                                return []

                                if not hasattr(genai, "embeddings") or not hasattr(
                                                genai.embeddings, "create"
                                ):
                                                raise ProviderRequestError(
                                                                "Embedding API not available in google-generativeai SDK"
                                                )

                                kwargs = {}
                                if self.embedding_model:
                                                kwargs["model"] = self.embedding_model

                                # ensure inputs is a list (SDKs commonly accept a single string or
                                # list)
                                payload = list(inputs)

                                try:
                                                result = genai.embeddings.create(input=payload, **kwargs)
                                                data = getattr(result, "data", None) or (
                                                                result.get("data") if isinstance(result, dict) else None
                                                )
                                                if not data:
                                                                raise ProviderRequestError("Embedding API returned no data")

                                                vectors: List[List[float]] = []
                                                for item in data:
                                                                if isinstance(item, dict):
                                                                                emb = item.get("embedding")
                                                                else:
                                                                                emb = getattr(item, "embedding", None)
                                                                if emb is None:
                                                                        raise ProviderRequestError(
                                                                                "Embedding item missing 'embedding' vector"
                                                                        )
                                                                vectors.append(list(emb))
                                                return vectors
                                except ProviderRequestError:
                                                raise
                                except Exception as exc:
                                                raise ProviderRequestError(f"Failed to create embeddings: {exc}") from exc

                # -----------------------
                # Indexing and querying
                # -----------------------
                def index_document(
                                self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
                ) -> None:
                                chunks = self.chunk_text(text)
                                if not chunks:
                                                return

                                vectors = self._create_embeddings(chunks)
                                if len(vectors) != len(chunks):
                                                raise ProviderRequestError(
                                                        "Embedding count does not match chunk count"
                                                )

                                chunk_items: List[Dict[str, Any]] = []
                                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                                                chunk_items.append(
                                                                {
                                                                        "id": f"{doc_id}::chunk::{i}",
                                                                        "vector": vec,
                                                                        "metadata": metadata or {},
                                                                        "text": chunk,
                                                                }
                                                )

                                try:
                                        self.vector_db.upsert(chunk_items)
                                except Exception as exc:
                                                raise ProviderRequestError(
                                                                f"Failed to upsert vectors to vector DB: {exc}"
                                                ) from exc

                def query_vector_db(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
                                try:
                                                q_vecs = self._create_embeddings([query])
                                                if not q_vecs or len(q_vecs) != 1:
                                                        raise ProviderRequestError(
                                                                "Failed to create embedding for query"
                                                        )
                                                q_vec = q_vecs[0]
                                except ProviderRequestError:
                                                raise
                                except Exception as exc:
                                                raise ProviderRequestError(f"Query embedding failed: {exc}") from exc

                                try:
                                                return self.vector_db.query(q_vec, top_k=top_k)
                                except Exception as exc:
                                                raise ProviderRequestError(f"Vector DB query failed: {exc}") from exc

                def retrieve_relevant_context(
                                self, query: str, top_k: int = 3, joiner: str = "\n---\n"
                ) -> str:
                                results = self.query_vector_db(query, top_k=top_k)
                                if not results:
                                        return ""
                                texts = [res.get("text", "") for res in results if res.get("text")]
                                return joiner.join(texts)

                # -----------------------
                # High-level RAG send
                # -----------------------
                def send_with_rag(
                                self,
                                prompt: str,
                                top_k: int = 3,
                                prepend_context: bool = True,
                                context_prefix: Optional[str] = None,
                ) -> str:
                                if not self.embedding_model:
                                                raise ProviderRequestError(
                                                        "Embedding model not configured; cannot perform RAG. "
                                                        "Provide `embedding_model` when creating the provider."
                                                )

                                context = self.retrieve_relevant_context(prompt, top_k=top_k)
                                if context:
                                                prefix = (
                                                                context_prefix
                                                                if context_prefix is not None
                                                                else "Context (retrieved):"
                                                )
                                                if prepend_context:
                                                        combined = f"{prefix}\n{context}\n\n{prompt}"
                                                else:
                                                        combined = f"{prompt}\n\n{prefix}\n{context}"
                                else:
                                                combined = prompt

                                return self._send_impl(combined)
