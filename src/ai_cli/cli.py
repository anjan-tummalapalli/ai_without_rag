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
from typing import TYPE_CHECKING, Any
 
from ai_cli.core.api import ask
from ai_cli.providers.bootstrap import init_providers
 
if TYPE_CHECKING:
    from ai_cli.rag.pipeline import RAGPipeline
 
VERSION = "0.3.0"
 
# Maximum number of bytes accepted from stdin before truncation.
_STDIN_MAX_BYTES: int = 102_400  # 100 KiB — same ceiling as the prompt char limit
 
# Hard ceiling on exponential-backoff sleep to prevent accidental DoS-in-place.
_MAX_BACKOFF_SECONDS: float = 30.0
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ai_cli")
 
 
def _sanitize_log_value(value: object) -> str:
    """
    Strip newlines and carriage-returns from a value before it enters a log
    message.  Prevents log-injection attacks where a crafted provider/model
    string could forge fake log lines.
    """
    return str(value).replace("\n", "\\n").replace("\r", "\\r")
 
 
def _safe_resolve_path(raw: str | None) -> str | None:
    """
    Resolve *raw* to an absolute, canonical path and reject obvious traversal
    attempts.
 
    Returns the resolved path string on success, or ``None`` when the input
    looks like a path-traversal attempt (e.g. contains ``..`` components or
    null bytes).  Callers must treat a ``None`` return as an error.
 
    Note: this does **not** restrict reads to a specific directory — the CLI is
    a developer tool and intentionally allows arbitrary file paths.  The check
    exists solely to surface accidental or malicious traversal sequences so
    they can be rejected early with a clear error message rather than silently
    opening an unexpected file.
    """
    if raw is None:
        return None
    
    if "\x00" in raw:
        return None
    # os.path.realpath collapses ".." and symlinks; we then check that the
    # normalised form does not differ from what a naive join would produce
    # in a way that suggests hidden traversal.
    resolved = os.path.realpath(raw)
    # Reject if the raw path contained explicit traversal sequences
    if ".." in raw.split(os.sep):
        return None
    return resolved
 
 
def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-cli",
        description="Enterprise AI CLI Gateway for multi-provider AI interactions.",
    )
    parser.add_argument(
        "-p", "--provider", default="auto", type=str,
        help="AI provider name (default: auto).",
    )
    parser.add_argument(
        "-q", "--prompt", type=str,
        help="Prompt/question to send to the AI provider.",
    )
    parser.add_argument(
        "-m", "--model", default=None, type=str,
        help="Optional model override for the selected provider.",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Start an interactive REPL chat loop.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Request timeout in seconds. Default: 60",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Profile name or configuration to use (optional).",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Enable streaming responses if supported.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}",
        help="Show program version and exit.",
    )
    parser.add_argument(
        "--rag", action="store_true",
        help="Enable RAG retrieval for the prompt.",
    )
    parser.add_argument(
        "--rag-docs", nargs="*",
        help="Documents to index into the RAG store (file paths or raw text).",
    )
    parser.add_argument(
        "--rag-chunk-size", type=int, default=500,
        help="Chunk size (characters) for RAG chunking (default 500).",
    )
    parser.add_argument(
        "--rag-chunk-overlap", type=int, default=50,
        help="Overlap (characters) between chunks (default 50).",
    )
    parser.add_argument(
        "--rag-top-k", type=int, default=5,
        help="Number of top chunks to retrieve for context (default 5).",
    )
    parser.add_argument(
        "--modules", type=str, default=None,
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
    profile = profile.strip() if isinstance(profile, str) and profile.strip() else None
 
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
        # Omit None values; include only keys accepted by ask()
        filtered: dict[str, Any] = {
            k: v
            for k, v in candidate.items()
            if v is not None and (accepts_var_kw or k in params)
        }
        # Guarantee required positional keys survive even if they were None
        filtered.setdefault("prompt", prompt)
        filtered.setdefault("provider", provider)
        return filtered
    except (TypeError, ValueError):
        logger.debug("Failed to inspect ask() signature; sending best-effort args")
        base: dict[str, Any] = {"provider": provider, "prompt": prompt, "timeout": timeout}
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
    """Decode a streaming chunk to a UTF-8 string."""
    if isinstance(chunk, bytes):
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
    Prints streaming pieces if iterable; otherwise prints the final value.
    """
    try:
        # Unwrap a coroutine first, then treat the resolved value uniformly
        value = await result if inspect.isawaitable(result) else result
        if isinstance(value, AsyncIterable):
            async for part in value:
                print(_decode_chunk(part), end="", flush=True)
            print()
            return 0
        if isinstance(value, Iterable) and not isinstance(value, str | bytes | dict):
            for part in value:
                print(_decode_chunk(part), end="", flush=True)
            print()
            return 0
        print(_decode_chunk(value))
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
    Delegates awaitable / async-iterable results to the async handler.
    """
    try:
        if inspect.isawaitable(result) or isinstance(result, AsyncIterable):
            return asyncio.run(_drain_async_result(result))
        if isinstance(result, Iterable) and not isinstance(result, str | bytes | dict):
            for part in result:
                print(_decode_chunk(part), end="", flush=True)
            print()
            return 0
        if isinstance(result, dict | list):
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0
        print(_decode_chunk(result))
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
    Invoke ask() with exponential-backoff retries on transient errors.
    Handles sync and async responses transparently. Returns an exit code.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
 
    for attempt in range(1, max_retries + 1):
        try:
            safe_kwargs = {
                k: ("<redacted>" if k == "prompt" else _sanitize_log_value(v))
                for k, v in kwargs.items()
            }
            logger.debug("Calling ask() attempt %d with kwargs=%s", attempt, safe_kwargs)
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
            sleep_secs = min(backoff * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
            time.sleep(sleep_secs)
        except KeyboardInterrupt:
            logger.warning("operation interrupted by user")
            return 130
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.exception("ai request failed: %s", exc)
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
 
    # Unreachable: loop always returns, but satisfies type checkers.
    return 1  # pragma: no cover
 
 
_HELP_TEXT = "\n".join([
    "Commands:",
    "  /index <file|text>      Index a file path or raw text into RAG",
    "  /search <query>         Retrieve top-k context from RAG",
    "  /switch <provider>      Switch provider",
    "  /model <model>          Set model override",
    "  /profile <name>         Set profile",
    "  /stream                 Toggle streaming mode",
    "  /exit, /quit            Exit",
])
 
_RAG_PROMPT_TEMPLATE = (
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}"
)
 
 
def run_interactive(
    provider: str,
    model: str | None,
    timeout: int,
    profile: str | None = None,
    stream: bool = False,
    rag: RAGPipeline | None = None,
    rag_chunk_size: int = 500,
    rag_chunk_overlap: int = 50,
    rag_top_k: int = 5,
    modules: str | None = None,
) -> int:
    """Run an interactive chat session."""
    from ai_cli.rag.pipeline import (
        RAGPipeline as _RAGPipeline,  # local import avoids circular
    )
 
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
    # Allow interactive RAG commands (/index, /search) even without --rag at startup.
    pipeline: RAGPipeline = rag if rag is not None else _RAGPipeline(embed_dim=128)
 
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return 0
 
        if not user_input:
            continue
 
        # --- command dispatch ---
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            return 0
 
        if user_input.lower() == "/help":
            print(_HELP_TEXT)
            continue
 
        if user_input.startswith("/switch"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                current_provider = parts[1].strip()
                print(f"Switched provider to: {current_provider}")
            else:
                print("Usage: /switch <provider>", file=sys.stderr)
            continue
 
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                current_model = parts[1].strip()
                print(f"Model set to: {current_model}")
            else:
                current_model = None
                print("Model cleared; using provider default.")
            continue
 
        if user_input.startswith("/profile"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                current_profile = parts[1].strip()
                print(f"Profile set to: {current_profile}")
            else:
                current_profile = None
                print("Profile cleared; using default.")
            continue
 
        if user_input.startswith("/stream"):
            current_stream = not current_stream
            print(f"Streaming {'enabled' if current_stream else 'disabled'}.")
            continue
 
        if user_input.startswith("/index"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                payload = parts[1].strip()
                if os.path.exists(payload):
                    safe_path = _safe_resolve_path(payload)
                    if safe_path is None:
                        print(
                            "Error: path contains illegal traversal sequences.",
                            file=sys.stderr,
                        )
                    else:
                        try:
                            with open(safe_path, encoding="utf-8") as fh:
                                text = fh.read()
                            pipeline.upsert_documents(
                                [text],
                                doc_ids=[safe_path],
                                chunk_size=rag_chunk_size,
                                overlap=rag_chunk_overlap,
                            )
                            print(f"Indexed file: {safe_path}")
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
 
        if user_input.startswith("/search"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                query = parts[1].strip()
                ctx = pipeline.retrieve_context(query, top_k=rag_top_k)
                print("\n--- RAG Context ---\n")
                print(ctx or "(no context indexed)")
                print("\n--- End Context ---\n")
            else:
                print("Usage: /search <query>", file=sys.stderr)
            continue
 
        # Normal message: optionally augment with RAG context
        context = pipeline.retrieve_context(user_input, top_k=rag_top_k)
        used_prompt = (
            _RAG_PROMPT_TEMPLATE.format(context=context, question=user_input)
            if context
            else user_input
        )
        # Fix 5 (prompt size): cap after RAG expansion to prevent oversized payloads
        if len(used_prompt) > 100_000:
            print(
                "Warning: prompt is very large after RAG expansion; truncating to 100k characters.",
                file=sys.stderr,
            )
            used_prompt = used_prompt[:100_000]
 
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
 
    return 0  # unreachable but satisfies type checkers
 
 
def _read_stdin_prompt() -> str:
    """
    Read and decode a prompt from piped stdin.
 
    Reads at most ``_STDIN_MAX_BYTES`` bytes before decoding to prevent
    unbounded memory allocation from an oversized pipe.  Returns an empty
    string on EOF.
    """
    raw = sys.stdin.buffer.read(_STDIN_MAX_BYTES)
    if not raw:
        return ""
    # Peek to see if the stream had more data — warn the user if so.
    leftover = sys.stdin.buffer.read(1)
    if leftover:
        print(
            f"Warning: stdin input exceeded {_STDIN_MAX_BYTES} bytes; "
            "truncating to limit.",
            file=sys.stderr,
        )
    return raw.decode("utf-8", errors="replace").strip()
 
 
def _load_rag_docs(
    rag_docs: list[str],
    rag_chunk_size: int,
    rag_chunk_overlap: int,
) -> RAGPipeline:
    """Instantiate a RAGPipeline and index the provided file paths / raw texts."""
    from ai_cli.rag.pipeline import RAGPipeline as _RAGPipeline
 
    pipeline = _RAGPipeline(embed_dim=128)
    docs: list[str] = []
    doc_ids: list[str] = []
 
    for payload in rag_docs:
        if os.path.exists(payload):
            safe_path = _safe_resolve_path(payload)
            if safe_path is None:
                logger.warning(
                    "Skipping %r: path contains illegal traversal sequences.", payload
                )
                continue
            try:
                with open(safe_path, encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeError) as exc:
                logger.warning("Failed to read %s: %s", safe_path, exc)
                continue
            docs.append(text)
            doc_ids.append(safe_path)
        else:
            docs.append(payload)
            doc_ids.append(str(uuid.uuid4()))
 
    if docs:
        pipeline.upsert_documents(
            docs,
            doc_ids=doc_ids,
            chunk_size=rag_chunk_size,
            overlap=rag_chunk_overlap,
        )
        logger.info("Indexed %d documents (%d chunks)", len(docs), len(pipeline))
 
    return pipeline
 
 
def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    init_providers()
    parser = build_parser()
    args = parser.parse_args(argv)
 
    if args.debug:
        logger.setLevel(logging.DEBUG)
 
    # argparse enforces type=int so only the range check is needed
    if args.timeout <= 0:
        parser.error("timeout must be a positive integer")
 
    # Initialise RAG pipeline when explicitly requested or docs are provided
    rag_pipeline: RAGPipeline | None = None
    if args.rag or args.rag_docs:
        rag_pipeline = _load_rag_docs(
            args.rag_docs or [],
            args.rag_chunk_size,
            args.rag_chunk_overlap,
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
 
    if not prompt:
        try:
            if not sys.stdin.isatty():
                prompt = _read_stdin_prompt()
                if not prompt:
                    parser.error("Prompt is required via --prompt or stdin.")
        except OSError as exc:
            logger.exception("failed to read stdin: %s", exc)
            print(f"ERROR: failed to read stdin: {exc}", file=sys.stderr)
            return 1
 
    if not prompt:
        parser.error("Prompt is required via --prompt or stdin.")
 
    if len(prompt) > 100_000:
        print("Warning: prompt is very large; truncating to 100k characters.", file=sys.stderr)
        prompt = prompt[:100_000]
 
    if args.rag and rag_pipeline is not None:
        context = rag_pipeline.retrieve_context(prompt, top_k=args.rag_top_k)
        if context:
            prompt = _RAG_PROMPT_TEMPLATE.format(context=context, question=prompt)
 
    logger.info(
        "provider=%s model=%s profile=%s stream=%s timeout=%s rag=%s",
        _sanitize_log_value(args.provider),
        _sanitize_log_value(args.model),
        _sanitize_log_value(args.profile),
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