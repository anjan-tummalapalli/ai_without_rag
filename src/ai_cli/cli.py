from __future__ import annotations
"""
ai_cli.cli - Command-line interface for the Enterprise AI CLI Gateway.

Enhancements (Advanced RAG)
- Chunking: configurable chunk size & overlap to break large documents into
        context windows for retrieval-augmented generation.
- Embedding: deterministic, lightweight embedding function (crypto-hash based)
        so the RAG pipeline is self-contained and has repeatable embeddings without
        external ML dependencies. Designed for demo/local usage; replace with model
        embeddings for production.
- Vector DB querying: an in-memory vector store with upsert and top-k cosine
        similarity retrieval. Stores chunk metadata and provides a retrieve_context()
        API used by the CLI to prepend context to prompts.
- CLI integration: flags --rag, --rag-docs, --rag-chunk-size,
        --rag-chunk-overlap, and --rag-top-k. REPL commands for indexing and search
        added: "index <file|text>", "search <query>".

Notes:
- The embedded vector store is intentionally lightweight and dependency-free.
        Swap in a production vector DB (FAISS, Milvus, Pinecone, etc.) and real
        embedding models (OpenAI, sentence-transformers) as needed.
"""
import argparse
import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import sys
import time
import uuid
from typing import Any, Iterable, AsyncIterable, Sequence, List, Dict, Optional

# The core ask() API is expected to be provided elsewhere in the package.
from ai_cli.core.api import ask

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
VERSION = "0.3.0"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
                stream=sys.stderr,
)
logger = logging.getLogger("ai_cli")


# -----------------------------------------------------------------------------
# Simple, self-contained RAG Pipeline implementation
# -----------------------------------------------------------------------------
class RAGPipeline:
                """
                Lightweight in-memory RAG pipeline:
                - chunk_texts: split documents into overlapping chunks
                - embed_texts: deterministic hash-based embeddings
                - upsert_documents: store chunk+embeddings in an in-memory vector DB
                - retrieve_context: top-k retrieval by cosine similarity
                """

                def __init__(self, embed_dim: int = 128):
                                self.embed_dim = embed_dim
                                # store entries as dicts: {id, doc_id, chunk, embedding, meta}
                                self._store: List[Dict[str, Any]] = []

                def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
                                """Chunk a text into overlapping windows of roughly chunk_size chars."""
                                if chunk_size <= 0:
                                                raise ValueError("chunk_size must be positive")
                                if overlap < 0:
                                                overlap = 0
                                texts = []
                                start = 0
                                L = len(text)
                                while start < L:
                                                end = min(start + chunk_size, L)
                                                chunk = text[start:end].strip()
                                                if chunk:
                                                                texts.append(chunk)
                                                if end == L:
                                                                break
                                                start = max(0, end - overlap)
                                return texts

                def _embed_one(self, text: str) -> List[float]:
                                """
                                Deterministic lightweight embedding using SHA256 with a counter to
                                produce embed_dim floats in [-1, 1]. Not a semantic embedding for
                                production use — swap in a model-based embedder for real applications.
                                """
                                vec = []
                                for i in range(self.embed_dim):
                                                h = hashlib.sha256()
                                                # mix in counter and text, produce stable bytes
                                                h.update(text.encode("utf-8"))
                                                h.update(i.to_bytes(2, "little"))
                                                digest = h.digest()
                                                # take first 4 bytes to make a 32-bit unsigned int
                                                val = int.from_bytes(digest[:4], "little")
                                                # scale to [-1, 1]
                                                vec.append((val / 0xFFFFFFFF) * 2.0 - 1.0)
                                return vec

                def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
                                return [self._embed_one(t) for t in texts]

                @staticmethod
                def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
                                # safe cosine similarity
                                num = 0.0
                                norma = 0.0
                                normb = 0.0
                                for x, y in zip(a, b):
                                                num += x * y
                                                norma += x * x
                                                normb += y * y
                                if norma <= 0 or normb <= 0:
                                                return 0.0
                                return num / (math.sqrt(norma) * math.sqrt(normb))

                def upsert_documents(
                                self,
                                doc_texts: Sequence[str],
                                doc_ids: Optional[Sequence[str]] = None,
                                chunk_size: int = 500,
                                overlap: int = 50,
                ) -> None:
                                """
                                Index/Upsert provided documents. Each document is chunked and embedded
                                and stored in the in-memory vector DB.
                                """
                                if doc_ids is None:
                                                doc_ids = [str(uuid.uuid4()) for _ in doc_texts]
                                for doc_id, text in zip(doc_ids, doc_texts):
                                                chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                                                embeddings = self.embed_texts(chunks)
                                                for chunk, emb in zip(chunks, embeddings):
                                                                entry = {
                                                                                "id": str(uuid.uuid4()),
                                                                                "doc_id": doc_id,
                                                                                "chunk": chunk,
                                                                                "embedding": emb,
                                                                                "meta": {"length": len(chunk)},
                                                                }
                                                                self._store.append(entry)
                                logger.info("Indexed %d documents -> %d chunks total", len(doc_texts), len(self._store))

                def retrieve_context(self, query: str, top_k: int = 5) -> str:
                                """Return the concatenated top-k chunks most similar to the query."""
                                if not self._store:
                                                return ""
                                q_emb = self._embed_one(query)
                                scored = []
                                for entry in self._store:
                                                score = self._cosine(q_emb, entry["embedding"])
                                                scored.append((score, entry))
                                scored.sort(key=lambda x: x[0], reverse=True)
                                top = [entry for score, entry in scored[:top_k] if score > 0.0]
                                # Join with separators; maintain order by score
                                context = "\n\n---\n\n".join(e["chunk"] for e in top)
                                return context


# -----------------------------------------------------------------------------
# CLI Argument Parsing
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
                """Build and return the CLI argument parser."""
                parser = argparse.ArgumentParser(
                                prog="ai-cli",
                                description=(
                                                "Enterprise AI CLI Gateway for multi-provider AI interactions."
                                ),
                )
                parser.add_argument(
                                "-p",
                                "--provider",
                                default="auto",
                                type=str,
                                help="AI provider name (default: auto).",
                )
                parser.add_argument(
                                "-q",
                                "--prompt",
                                type=str,
                                help="Prompt/question to send to the AI provider.",
                )
                parser.add_argument(
                                "-m",
                                "--model",
                                default=None,
                                type=str,
                                help="Optional model override for the selected provider.",
                )
                parser.add_argument(
                                "-i",
                                "--interactive",
                                action="store_true",
                                help="Start an interactive REPL chat loop.",
                )
                parser.add_argument(
                                "--timeout",
                                type=int,
                                default=60,
                                help="Request timeout in seconds. Default: 60",
                )
                parser.add_argument(
                                "--debug",
                                action="store_true",
                                help="Enable debug logging.",
                )
                parser.add_argument(
                                "--profile",
                                type=str,
                                default=None,
                                help="Profile name or configuration to use (optional).",
                )
                parser.add_argument(
                                "--stream",
                                action="store_true",
                                help="Enable streaming responses if supported.",
                )
                parser.add_argument(
                                "--version",
                                action="version",
                                version=f"%(prog)s {VERSION}",
                                help="Show program version and exit.",
                )
                parser.add_argument(
                                "--rag",
                                action="store_true",
                                help="Enable RAG retrieval for the prompt.",
                )
                parser.add_argument(
                                "--rag-docs",
                                nargs="*",
                                help="Documents to index into the RAG store (file paths or raw text).",
                )
                parser.add_argument(
                                "--rag-chunk-size",
                                type=int,
                                default=500,
                                help="Chunk size (characters) for RAG document chunking (default 500).",
                )
                parser.add_argument(
                                "--rag-chunk-overlap",
                                type=int,
                                default=50,
                                help="Overlap (characters) between chunks (default 50).",
                )
                parser.add_argument(
                                "--rag-top-k",
                                type=int,
                                default=5,
                                help="Number of top chunks to retrieve for context (default 5).",
                )
                return parser


# -----------------------------------------------------------------------------
# Helper to build kwargs for ask() robustly
# -----------------------------------------------------------------------------
def _build_ask_kwargs(
                provider: str,
                prompt: str,
                model: str | None,
                timeout: int,
                profile: str | None = None,
                stream: bool = False,
) -> dict[str, Any]:
                """
                Build the kwargs to pass to ask() based on what parameters it
                accepts. Uses inspect.signature so the CLI remains compatible with
                different ask() versions. Only keys accepted by ask() (or
                variable kwargs) are passed. None values are omitted.
                """
                # Normalize simple invalid inputs
                provider = provider.strip() if provider and provider.strip() else "auto"
                prompt = prompt or ""
                model = model.strip() if isinstance(model, str) and model.strip() else None
                profile = profile.strip() if isinstance(profile, str) and profile.strip() else None

                candidate = {
                                "provider": provider,
                                "prompt": prompt,
                                "model": model,
                                "timeout": timeout,
                                "profile": profile,
                                "stream": stream,
                }

                try:
                                sig = inspect.signature(ask)
                                params = sig.parameters
                                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                                # Only include keys that ask accepts or if it accepts **kwargs
                                filtered: dict[str, Any] = {}
                                for k, v in candidate.items():
                                                if v is None:
                                                                # omit None values to maximize compatibility
                                                                continue
                                                if accepts_var_kw or k in params:
                                                                filtered[k] = v
                                # Ensure mandatory keys are present if ask expects them
                                if "prompt" in params and "prompt" not in filtered:
                                                filtered["prompt"] = prompt
                                if "provider" in params and "provider" not in filtered:
                                                filtered["provider"] = provider
                                return filtered
                except Exception:
                                logger.debug("Failed to inspect ask() signature; sending best-effort args")
                                # Fallback: only base safe args
                                base: dict[str, Any] = {"provider": provider, "prompt": prompt, "timeout": timeout}
                                if model is not None:
                                                base["model"] = model
                                if profile is not None:
                                                base["profile"] = profile
                                if stream:
                                                base["stream"] = stream
                                return base


# -----------------------------------------------------------------------------
# Response handling (sync + async + streaming)
# -----------------------------------------------------------------------------
def _decode_chunk(chunk: Any) -> str:
                if isinstance(chunk, bytes):
                                try:
                                                return chunk.decode("utf-8")
                                except Exception:
                                                return chunk.decode("utf-8", errors="replace")
                if isinstance(chunk, str):
                                return chunk
                try:
                                return json.dumps(chunk, default=str, ensure_ascii=False)
                except Exception:
                                return str(chunk)


async def _drain_async_result(result: Any, provider: str, stream: bool) -> int:
                """
                Handle coroutine results, async iterables or async generators from
                ask(). Prints streaming pieces if iterable; otherwise prints final
                value.
                """
                try:
                                # If it's an async iterable (async generator), iterate
                                if isinstance(result, AsyncIterable):
                                                async for part in result:  # type: ignore
                                                                text = _decode_chunk(part)
                                                                print(text, end="", flush=True)
                                                # newline at end
                                                print()
                                                return 0
                                # If it's an awaitable coroutine that returns a value
                                value = await result
                                # If the awaited value is async iterable, handle it (rare)
                                if isinstance(value, AsyncIterable):
                                                async for part in value:  # type: ignore
                                                                print(_decode_chunk(part), end="", flush=True)
                                                print()
                                                return 0
                                # If the awaited value is an iterator/generator, iterate it
                                if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                                                for part in value:
                                                                print(_decode_chunk(part), end="", flush=True)
                                                print()
                                                return 0
                                # Scalar / dict / list etc.
                                text = _decode_chunk(value)
                                print(text)
                                return 0
                except KeyboardInterrupt:
                                print("\n[Interrupted]", file=sys.stderr)
                                return 130
                except Exception as exc:
                                logger.exception("async request failed: %s", exc)
                                print(f"ERROR: {exc}", file=sys.stderr)
                                return 1


def _handle_sync_result(result: Any, provider: str, stream: bool) -> int:
                """
                Handle synchronous results: scalars, bytes, dicts, iterables
                (generators).
                """
                try:
                                # Asyncables slipped through
                                if inspect.isawaitable(result):
                                                return asyncio.run(_drain_async_result(result, provider, stream))

                                # If it's an async iterable object instance (rare)
                                if isinstance(result, AsyncIterable):
                                                return asyncio.run(_drain_async_result(result, provider, stream))

                                # Iterable streaming (but strings are iterable of chars -> avoid)
                                if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
                                                for part in result:
                                                                print(_decode_chunk(part), end="", flush=True)
                                                print()
                                                return 0

                                # Bytes
                                if isinstance(result, bytes):
                                                try:
                                                                print(result.decode("utf-8"))
                                                except Exception:
                                                                print(result.decode("utf-8", errors="replace"))
                                                return 0

                                # dict / list / scalar
                                if isinstance(result, (dict, list)):
                                                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                                                return 0

                                print(str(result))
                                return 0
                except KeyboardInterrupt:
                                print("\n[Interrupted]", file=sys.stderr)
                                return 130
                except Exception as exc:
                                logger.exception("sync request failed: %s", exc)
                                print(f"ERROR: {exc}", file=sys.stderr)
                                return 1


# -----------------------------------------------------------------------------
# Retry wrapper for transient errors
# -----------------------------------------------------------------------------
def _invoke_with_retries(kwargs: dict[str, Any], max_retries: int = 3, backoff: float = 0.5) -> int:
                """
                Invoke ask() with a small retry/backoff on transient errors.
                Handles sync and async responses transparently. Returns an exit code.
                """
                attempt = 0
                while True:
                                try:
                                                attempt += 1
                                                # Prepare a safe-to-log copy of kwargs
                                                safe_kwargs = {k: ("<redacted>" if k == "prompt" else v) for k, v in kwargs.items()}
                                                logger.debug("Calling ask() attempt %d with kwargs=%s", attempt, safe_kwargs)
                                                result = ask(**kwargs)
                                                # If the result is awaitable or async iterable, handle via asyncio
                                                if inspect.isawaitable(result) or isinstance(result, AsyncIterable):
                                                                return asyncio.run(
                                                                                _drain_async_result(
                                                                                                result, kwargs.get("provider", "unknown"), kwargs.get("stream", False)
                                                                                )
                                                                )
                                                # Otherwise handle sync
                                                return _handle_sync_result(result, kwargs.get("provider", "unknown"), kwargs.get("stream", False))
                                except (TimeoutError, ConnectionError, OSError) as exc:
                                                logger.warning("Transient error on attempt %d: %s", attempt, exc)
                                                if attempt >= max_retries:
                                                                logger.error("Max retries reached (%d). Failing.", max_retries)
                                                                print(f"ERROR: {exc}", file=sys.stderr)
                                                                return 124 if isinstance(exc, TimeoutError) else 1
                                                sleep = backoff * (2 ** (attempt - 1))
                                                time.sleep(sleep)
                                                continue
                                except KeyboardInterrupt:
                                                logger.warning("operation interrupted by user")
                                                return 130
                                except Exception as exc:
                                                logger.exception("ai request failed: %s", exc)
                                                print(f"ERROR: {exc}", file=sys.stderr)
                                                return 1


# -----------------------------------------------------------------------------
# Interactive REPL Loop
# -----------------------------------------------------------------------------
def run_interactive(
                provider: str,
                model: str | None,
                timeout: int,
                profile: str | None = None,
                stream: bool = False,
                rag: Optional[RAGPipeline] = None,
                rag_chunk_size: int = 500,
                rag_chunk_overlap: int = 50,
                rag_top_k: int = 5,
) -> int:
                """Run an interactive chat session."""
                print("--- AI CLI Interactive Mode ---")
                print(
                                f"Provider: {provider} | Model: {model or 'default'} | "
                                f"Profile: {profile or 'default'} | Stream: {stream}"
                )
                print(
                                "Type /switch <provider>, /model <model>, /profile <name>, "
                                "/stream, /index <file|text>, /search <query>, /exit or /quit. Type /help for this text.\n"
                )

                current_provider = provider
                current_model = model
                current_profile = profile
                current_stream = stream
                pipeline = rag

                while True:
                                try:
                                                user_input = input("\nYou: ").strip()
                                except (EOFError, KeyboardInterrupt):
                                                print("\nExiting...")
                                                return 0

                                if not user_input:
                                                continue

                                cmd = user_input.strip()
                                if cmd.lower() in ("/exit", "/quit", "exit", "quit"):
                                                print("Goodbye!")
                                                return 0

                                if cmd.lower() in ("/help",):
                                                print("Commands:")
                                                print("  /index <file|text>      Index a file path or raw text into the RAG store")
                                                print("  /search <query>         Retrieve top-k context from RAG and print it")
                                                print("  /switch <provider>      Switch provider")
                                                print("  /model <model>          Set model override")
                                                print("  /profile <name>         Set profile")
                                                print("  /stream                 Toggle streaming mode")
                                                print("  /exit, /quit            Exit")
                                                continue

                                if cmd.startswith("/switch"):
                                                parts = cmd.split(maxsplit=1)
                                                if len(parts) == 2 and parts[1].strip():
                                                                current_provider = parts[1].strip()
                                                                print(f"Switched provider to: {current_provider}")
                                                else:
                                                                print("Usage: /switch <provider>", file=sys.stderr)
                                                continue

                                if cmd.startswith("/model"):
                                                parts = cmd.split(maxsplit=1)
                                                if len(parts) == 2 and parts[1].strip():
                                                                current_model = parts[1].strip()
                                                                print(f"Model set to: {current_model}")
                                                else:
                                                                current_model = None
                                                                print("Model cleared; using provider default.")
                                                continue

                                if cmd.startswith("/profile"):
                                                parts = cmd.split(maxsplit=1)
                                                if len(parts) == 2 and parts[1].strip():
                                                                current_profile = parts[1].strip()
                                                                print(f"Profile set to: {current_profile}")
                                                else:
                                                                current_profile = None
                                                                print("Profile cleared; using default.")
                                                continue

                                if cmd.startswith("/stream"):
                                                current_stream = not current_stream
                                                print(f"Streaming {'enabled' if current_stream else 'disabled'}.")
                                                continue

                                if cmd.startswith("/index"):
                                                parts = cmd.split(maxsplit=1)
                                                if len(parts) == 2 and parts[1].strip():
                                                                payload = parts[1].strip()
                                                                # If it's a file path, read; otherwise treat as raw text
                                                                if os.path.exists(payload):
                                                                                try:
                                                                                                with open(payload, "r", encoding="utf-8") as fh:
                                                                                                                text = fh.read()
                                                                                                pipeline.upsert_documents([text], doc_ids=[payload], chunk_size=rag_chunk_size, overlap=rag_chunk_overlap)
                                                                                                print(f"Indexed file: {payload}")
                                                                                except Exception as exc:
                                                                                                print(f"Failed to index file: {exc}", file=sys.stderr)
                                                                else:
                                                                                pipeline.upsert_documents([payload], chunk_size=rag_chunk_size, overlap=rag_chunk_overlap)
                                                                                print("Indexed raw text provided.")
                                                else:
                                                                print("Usage: /index <file-path-or-raw-text>", file=sys.stderr)
                                                continue

                                if cmd.startswith("/search"):
                                                parts = cmd.split(maxsplit=1)
                                                if len(parts) == 2 and parts[1].strip():
                                                                query = parts[1].strip()
                                                                ctx = pipeline.retrieve_context(query, top_k=rag_top_k)
                                                                print("\n--- RAG Context ---\n")
                                                                print(ctx or "(no context indexed)")
                                                                print("\n--- End Context ---\n")
                                                else:
                                                                print("Usage: /search <query>", file=sys.stderr)
                                                continue

                                # Normal message => optionally augment with RAG
                                used_prompt = cmd
                                if pipeline is not None:
                                                context = pipeline.retrieve_context(cmd, top_k=rag_top_k)
                                                if context:
                                                                used_prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{cmd}"

                                print(f"[{current_provider}] Thinking...")
                                kwargs = _build_ask_kwargs(
                                                provider=current_provider,
                                                prompt=used_prompt,
                                                model=current_model,
                                                timeout=timeout,
                                                profile=current_profile,
                                                stream=current_stream,
                                )
                                exit_code = _invoke_with_retries(kwargs)
                                # continue interactive unless fatal
                                if exit_code not in (0, 130):
                                                print(f"[ERROR] command failed with code {exit_code}", file=sys.stderr)
                                continue

                return 0


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
                """Main CLI entrypoint."""
                parser = build_parser()
                args = parser.parse_args(argv)

                if args.debug:
                                logger.setLevel(logging.DEBUG)

                # Basic validation
                if args.timeout is None or not isinstance(args.timeout, int) or args.timeout <= 0:
                                parser.error("timeout must be a positive integer")
                if args.model is not None and not isinstance(args.model, str):
                                parser.error("model must be a string")
                if args.profile is not None and not isinstance(args.profile, str):
                                parser.error("profile must be a string")

                # Initialize RAG pipeline if requested (or if rag-docs provided)
                rag_pipeline: Optional[RAGPipeline] = None
                if args.rag or args.rag_docs:
                                rag_pipeline = RAGPipeline(embed_dim=128)

                # Index documents provided on the command line (file paths or raw text)
                if args.rag_docs and rag_pipeline is not None:
                                docs = []
                                doc_ids = []
                                for payload in args.rag_docs:
                                                if os.path.exists(payload):
                                                                try:
                                                                                with open(payload, "r", encoding="utf-8") as fh:
                                                                                                text = fh.read()
                                                                except Exception as exc:
                                                                                logger.warning("Failed to read %s: %s", payload, exc)
                                                                                text = ""
                                                                docs.append(text)
                                                                doc_ids.append(payload)
                                                else:
                                                                # treat as raw text
                                                                docs.append(payload)
                                                                doc_ids.append(str(uuid.uuid4()))
                                if docs:
                                                rag_pipeline.upsert_documents(docs, doc_ids=doc_ids, chunk_size=args.rag_chunk_size, overlap=args.rag_chunk_overlap)

                if args.interactive:
                                return run_interactive(
                                                args.provider,
                                                args.model,
                                                args.timeout,
                                                profile=args.profile,
                                                stream=args.stream,
                                                rag=rag_pipeline,
                                                rag_chunk_size=args.rag_chunk_size,
                                                rag_chunk_overlap=args.rag_chunk_overlap,
                                                rag_top_k=args.rag_top_k,
                                )

                prompt = args.prompt

                # Read from stdin only when piped
                if not prompt:
                                try:
                                                if not sys.stdin.isatty():
                                                                # read all piped stdin
                                                                raw = sys.stdin.buffer.read()
                                                                if not raw:
                                                                                parser.error("Prompt is required via --prompt or stdin.")
                                                                try:
                                                                                prompt = raw.decode("utf-8").strip()
                                                                except Exception:
                                                                                prompt = raw.decode("utf-8", errors="replace").strip()
                                                else:
                                                                parser.error(
                                                                                "Prompt is required via --prompt or stdin (piped). "
                                                                                "Or use -i for interactive mode."
                                                                )
                                except Exception as exc:
                                                logger.exception("failed to read stdin: %s", exc)
                                                print(f"ERROR: failed to read stdin: {exc}", file=sys.stderr)
                                                return 1

                if not prompt:
                                parser.error("Prompt is required via --prompt or stdin.")

                # safety checks on prompt size
                if len(prompt) > 100_000:
                                print(
                                                "Warning: prompt is very large; truncating to 100k "
                                                "characters.",
                                                file=sys.stderr,
                                )
                                prompt = prompt[:100_000]

                # If RAG is enabled, retrieve context and prepend to the prompt
                if args.rag and rag_pipeline is not None:
                                context = rag_pipeline.retrieve_context(prompt, top_k=args.rag_top_k)
                                if context:
                                                prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{prompt}"

                logger.info(
                                "provider=%s model=%s profile=%s stream=%s timeout=%s rag=%s",
                                args.provider,
                                args.model,
                                args.profile,
                                args.stream,
                                args.timeout,
                                bool(args.rag),
                )

                kwargs = _build_ask_kwargs(
                                provider=args.provider,
                                prompt=prompt,
                                model=args.model,
                                timeout=args.timeout,
                                profile=args.profile,
                                stream=args.stream,
                )

                return _invoke_with_retries(kwargs)


if __name__ == "__main__":
                raise SystemExit(main())
