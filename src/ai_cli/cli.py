from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterable, Iterable, Sequence
from typing import Any

from ai_cli.core.api import ask
from ai_cli.providers.bootstrap import init_providers
from ai_cli.rag.in_memory import InMemoryRAGPipeline

VERSION = "0.3.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ai_cli")

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
            prog="ai-cli",
            description="Enterprise AI CLI Gateway for multi-provider AI "
            "interactions.",
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
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
            help="Documents to index into the RAG store "
            "(file paths or raw text).",
    )
    parser.add_argument(
            "--rag-chunk-size",
            type=int,
            default=500,
            help="Chunk size (characters) for RAG chunking (default 500).",
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
    parser.add_argument(
            "--modules",
            type=str,
            default=None,
            help="Comma-separated list of modules to enable (e.g. mod1,mod2).",
    )
    return parser


def _build_ask_kwargs(
    provider: str,
    prompt: str,
    model: str | None,
    timeout: int,
    profile: str | None = None,
    stream: bool = False,
    modules: str | None = None,
) -> dict[str, Any]:
    """
    Build the kwargs to pass to ask() based on what parameters it accepts.
    Uses inspect.signature so the CLI remains compatible with different
    ask() versions. Only keys accepted by ask() (or variable kwargs) are
    passed. None values are omitted.
    """
    provider = provider.strip() if provider and provider.strip() else "auto"
    prompt = prompt or ""
    model = model.strip() if isinstance(model, str) and model.strip() else None
    profile = (
            profile.strip() if isinstance(profile, str) and profile.strip() else None
    )

    modules_list: list[str] | None = None
    if isinstance(modules, str) and modules.strip():
            modules_list = [m.strip() for m in modules.split(",") if m.strip()]

    candidate: dict[str, Any] = {
            "provider": provider,
            "prompt": prompt,
            "model": model,
            "timeout": timeout,
            "profile": profile,
            "stream": stream,
            "modules": modules_list,
    }

    try:
            sig = inspect.signature(ask)
            params = sig.parameters
            accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            filtered: dict[str, Any] = {}
            for k, v in candidate.items():
                    if v is None:
                            # omit None values to maximize compatibility
                            continue
                    if accepts_var_kw or k in params:
                            filtered[k] = v
            if "prompt" in params and "prompt" not in filtered:
                    filtered["prompt"] = prompt
            if "provider" in params and "provider" not in filtered:
                    filtered["provider"] = provider
            return filtered
    except (TypeError, ValueError):
            logger.debug("Failed to inspect ask() signature; sending best-effort args")
            base: dict[str, Any] = {
                    "provider": provider,
                    "prompt": prompt,
                    "timeout": timeout,
            }
            if model is not None:
                    base["model"] = model
            if profile is not None:
                    base["profile"] = profile
            if stream:
                    base["stream"] = stream
            if modules_list is not None:
                    base["modules"] = modules_list
            return base


def _decode_chunk(chunk: Any) -> str:
    if isinstance(chunk, bytes):
            try:
                    return chunk.decode("utf-8")
            except UnicodeDecodeError:
                    return chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, str):
            return chunk
    try:
            return json.dumps(chunk, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
            return str(chunk)


async def _drain_async_result(result: Any) -> int:
    """
    Handle coroutine results, async iterables or async generators from ask().
    Prints streaming pieces if iterable; otherwise prints final value.
    """
    try:
            if isinstance(result, AsyncIterable):
                    async for part in result:
                            text = _decode_chunk(part)
                            print(text, end="", flush=True)
                    print()
                    return 0
            value = await result
            if isinstance(value, AsyncIterable):
                    async for part in value:
                            print(_decode_chunk(part), end="", flush=True)
                    print()
                    return 0
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                    for part in value:
                            print(_decode_chunk(part), end="", flush=True)
                    print()
                    return 0
            text = _decode_chunk(value)
            print(text)
            return 0
    except KeyboardInterrupt:
            print("\n[Interrupted]", file=sys.stderr)
            return 130
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.exception("async request failed: %s", exc)
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1


def _handle_sync_result(result: Any) -> int:
    """
    Handle synchronous results: scalars, bytes, dicts, iterables (generators).
    """
    try:
            if inspect.isawaitable(result):
                    return asyncio.run(_drain_async_result(result))
            if isinstance(result, AsyncIterable):
                    return asyncio.run(_drain_async_result(result))
            if isinstance(result, Iterable) and not isinstance(
                    result, (str, bytes, dict)
            ):
                    for part in result:
                            print(_decode_chunk(part), end="", flush=True)
                    print()
                    return 0
            if isinstance(result, bytes):
                    try:
                            print(result.decode("utf-8"))
                    except UnicodeDecodeError:
                            print(result.decode("utf-8", errors="replace"))
                    return 0
            if isinstance(result, (dict, list)):
                    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                    return 0
            print(str(result))
            return 0
    except KeyboardInterrupt:
            print("\n[Interrupted]", file=sys.stderr)
            return 130
    except (RuntimeError, TypeError, ValueError) as exc:
            logger.exception("sync request failed: %s", exc)
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1


def _invoke_with_retries(
    kwargs: dict[str, Any], max_retries: int = 3, backoff: float = 0.5
) -> int:
    """
    Invoke ask() with a small retry/backoff on transient errors.
    Handles sync and async responses transparently. Returns an exit code.
    """
    if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

    for attempt in range(1, max_retries + 1):
            try:
                    safe_kwargs = {
                            k: ("<redacted>" if k == "prompt" else v)
                            for k, v in kwargs.items()
                    }
                    logger.debug(
                            "Calling ask() attempt %d with kwargs=%s", attempt, safe_kwargs
                    )
                    result = ask(**kwargs)
                    if inspect.isawaitable(result) or isinstance(result, AsyncIterable):
                            return asyncio.run(_drain_async_result(result))
                    return _handle_sync_result(result)
            except (TimeoutError, ConnectionError, OSError) as exc:
                    logger.warning("Transient error on attempt %d: %s", attempt, exc)
                    if attempt == max_retries:
                            logger.error("Max retries reached (%d). Failing.", max_retries)
                            print(f"ERROR: {exc}", file=sys.stderr)
                            return 124 if isinstance(exc, TimeoutError) else 1
                    time.sleep(backoff * (2 ** (attempt - 1)))
            except KeyboardInterrupt:
                    logger.warning("operation interrupted by user")
                    return 130
            except (RuntimeError, TypeError, ValueError) as exc:
                    logger.exception("ai request failed: %s", exc)
                    print(f"ERROR: {exc}", file=sys.stderr)
                    return 1


def run_interactive(
    provider: str,
    model: str | None,
    timeout: int,
    profile: str | None = None,
    stream: bool = False,
    rag: InMemoryRAGPipeline | None = None,
    rag_chunk_size: int = 500,
    rag_chunk_overlap: int = 50,
    rag_top_k: int = 5,
    modules: str | None = None,
) -> int:
    """Run an interactive chat session."""
    print("--- AI CLI Interactive Mode ---")
    print(
            f"Provider: {provider} | Model: {model or 'default'} | "
            f"Profile: {profile or 'default'} | Stream: {stream}"
    )
    print(
            "Type /switch <provider>, /model <model>, /profile <name>, "
            "/stream, /index <file|text>, /search <query>, /exit or /quit. "
            "Type /help for this text.\n"
    )

    current_provider = provider
    current_model = model
    current_profile = profile
    current_stream = stream
    pipeline = rag
    if pipeline is None:
            # Allow interactive RAG commands (/index, /search) even when
            # --rag was not supplied at startup.
            pipeline = InMemoryRAGPipeline(embed_dim=128)

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
                    print("  /index <file|text>      Index a file path or raw text into RAG")
                    print("  /search <query>         Retrieve top-k context from RAG")
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
                    print(
                            f"Streaming {'enabled' if current_stream else 'disabled'}."
                    )
                    continue
            if cmd.startswith("/index"):
                    parts = cmd.split(maxsplit=1)
                    if len(parts) == 2 and parts[1].strip():
                            payload = parts[1].strip()
                            # If it's a file path, read; otherwise treat as raw text
                            if os.path.exists(payload):
                                    try:
                                            with open(payload, encoding="utf-8") as fh:
                                                    text = fh.read()
                                            pipeline.upsert_documents(
                                                    [text],
                                                    doc_ids=[payload],
                                                    chunk_size=rag_chunk_size,
                                                    overlap=rag_chunk_overlap,
                                            )
                                            print(f"Indexed file: {payload}")
                                    except (OSError, UnicodeError) as exc:
                                            print(f"Failed to index file: {exc}", file=sys.stderr)
                            else:
                                    pipeline.upsert_documents(
                                            [payload],
                                            chunk_size=rag_chunk_size,
                                            overlap=rag_chunk_overlap,
                                    )
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
            context = pipeline.retrieve_context(cmd, top_k=rag_top_k)
            if context:
                    used_prompt = (
                            "Use the following context to answer the question.\n\n"
                            f"Context:\n{context}\n\nQuestion:\n{cmd}"
                    )

            print(f"[{current_provider}] Thinking...")
            kwargs = _build_ask_kwargs(
                    provider=current_provider,
                    prompt=used_prompt,
                    model=current_model,
                    timeout=timeout,
                    profile=current_profile,
                    stream=current_stream,
                    modules=modules,
            )
            exit_code = _invoke_with_retries(kwargs)
            if exit_code not in (0, 130):
                    print(f"[ERROR] command failed with code {exit_code}", file=sys.stderr)
            continue

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    init_providers()
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
    rag_pipeline: InMemoryRAGPipeline | None = None
    if args.rag or args.rag_docs:
            rag_pipeline = InMemoryRAGPipeline(embed_dim=128)

    # Index documents provided on the command line (file paths or raw text)
    if args.rag_docs and rag_pipeline is not None:
            docs: list[str] = []
            doc_ids: list[str] = []
            for payload in args.rag_docs:
                    if os.path.exists(payload):
                            try:
                                    with open(payload, encoding="utf-8") as fh:
                                            text = fh.read()
                            except (OSError, UnicodeError) as exc:
                                    logger.warning("Failed to read %s: %s", payload, exc)
                                    continue
                            docs.append(text)
                            doc_ids.append(payload)
                    else:
                            docs.append(payload)
                            doc_ids.append(str(uuid.uuid4()))
            if docs:
                    rag_pipeline.upsert_documents(
                            docs,
                            doc_ids=doc_ids,
                            chunk_size=args.rag_chunk_size,
                            overlap=args.rag_chunk_overlap,
                    )
                    logger.info(
                            "Indexed %d documents (%d chunks)",
                            len(docs),
                            len(rag_pipeline),
                    )

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
                    modules=args.modules,
            )

    prompt = args.prompt

    # Read from stdin only when piped
    if not prompt:
            try:
                    if not sys.stdin.isatty():
                            raw = sys.stdin.buffer.read()
                            if not raw:
                                    parser.error("Prompt is required via --prompt or stdin.")
                            try:
                                    prompt = raw.decode("utf-8").strip()
                            except UnicodeDecodeError:
                                    prompt = raw.decode("utf-8", errors="replace").strip()
            except (OSError, UnicodeError) as exc:
                    logger.exception("failed to read stdin: %s", exc)
                    print(f"ERROR: failed to read stdin: {exc}", file=sys.stderr)
                    return 1

    if not prompt:
            parser.error("Prompt is required via --prompt or stdin.")

    # safety checks on prompt size
    if len(prompt) > 100_000:
            print(
                    "Warning: prompt is very large; truncating to 100k characters.",
                    file=sys.stderr,
            )
            prompt = prompt[:100_000]

    # If RAG is enabled, retrieve context and prepend to the prompt
    if args.rag and rag_pipeline is not None:
            context = rag_pipeline.retrieve_context(prompt, top_k=args.rag_top_k)
            if context:
                    prompt = (
                            "Use the following context to answer the question.\n\n"
                            f"Context:\n{context}\n\nQuestion:\n{prompt}"
                    )

    logger.info(
            "provider=%s model=%s profile=%s stream=%s timeout=%s rag=%s",
            args.provider,
            args.model,
            args.profile,
            args.stream,
            args.timeout,
            rag_pipeline is not None,
    )

    kwargs = _build_ask_kwargs(
            provider=args.provider,
            prompt=prompt,
            model=args.model,
            timeout=args.timeout,
            profile=args.profile,
            stream=args.stream,
            modules=args.modules,
    )

    return _invoke_with_retries(kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
