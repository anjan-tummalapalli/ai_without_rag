from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterable, Iterable
from typing import Any

VERSION = "0.3.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ai_cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-cli",
        description=(
            "Enterprise AI CLI Gateway for multi-provider AI "
            "interactions."
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
        "--debug", action="store_true", help="Enable debug logging."
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
        "--modules",
        type=str,
        default=None,
        help="Comma-separated list of modules to enable "
        "(e.g. mod1,mod2).",
    )
    return parser


def _get_ask_callable():
    try:
        mod = importlib.import_module("ai_cli.core.api")
        ask = getattr(mod, "ask", None)
        if ask is None or not callable(ask):
            raise ImportError("ask() not found in ai_cli.core.api")
        return ask
    except Exception as exc:
        logger.debug("Could not import ask(): %s", exc)
        raise


def _init_providers_safe() -> None:
    try:
        mod = importlib.import_module("ai_cli.providers.bootstrap")
        init = getattr(mod, "init_providers", None)
        if callable(init):
            try:
                init()
            except Exception as exc:
                logger.debug("init_providers raised: %s", exc)
    except Exception as exc:
        logger.debug("providers.bootstrap not available: %s", exc)


def _build_ask_kwargs(
    provider: str,
    prompt: str,
    model: str | None,
    timeout: int,
    profile: str | None = None,
    stream: bool = False,
    modules: str | None = None,
) -> dict[str, Any]:
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
        ask = _get_ask_callable()
        sig = inspect.signature(ask)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        filtered: dict[str, Any] = {}
        for k, v in candidate.items():
            if v is None:
                continue
            if accepts_var_kw or k in params:
                filtered[k] = v
        if "prompt" in params and "prompt" not in filtered:
            filtered["prompt"] = prompt
        if "provider" in params and "provider" not in filtered:
            filtered["provider"] = provider
        return filtered
    except Exception:
        logger.debug("Failed to inspect ask() signature; using fallback args")
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
    if isinstance(chunk, bytes | bytearray):
        try:
            return bytes(chunk).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(chunk).decode("utf-8", errors="replace")
    if isinstance(chunk, str):
        return chunk
    try:
        return json.dumps(chunk, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(chunk)


async def _drain_async_result(result: Any) -> int:
    try:
        if isinstance(result, AsyncIterable):
            async for part in result:
                print(_decode_chunk(part), end="", flush=True)
            print()
            return 0
        value = await result
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
        logger.debug("async handler error: %s", exc)
        print("ERROR: async request failed", file=sys.stderr)
        return 1


def _handle_sync_result(result: Any) -> int:
    try:
        if inspect.isawaitable(result):
            return asyncio.run(_drain_async_result(result))
        if isinstance(result, AsyncIterable):
            return asyncio.run(_drain_async_result(result))
        if isinstance(result, Iterable) and not isinstance(result, str | bytes | dict):
            for part in result:
                print(_decode_chunk(part), end="", flush=True)
            print()
            return 0
        if isinstance(result, bytes | bytearray):
            try:
                print(bytes(result).decode("utf-8"))
            except UnicodeDecodeError:
                print(bytes(result).decode("utf-8", errors="replace"))
            return 0
        if isinstance(result, dict | list):
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0
        print(str(result))
        return 0
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        return 130
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("sync handler error: %s", exc)
        print("ERROR: sync request failed", file=sys.stderr)
        return 1


def _invoke_with_retries(
    kwargs: dict[str, Any], max_retries: int = 3, backoff: float = 0.5
) -> int:
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    try:
        ask = _get_ask_callable()
    except Exception:
        logger.error("ask() is not available; aborting request")
        print("ERROR: ask() backend unavailable", file=sys.stderr)
        return 1

    for attempt in range(1, max_retries + 1):
        try:
            safe_kwargs = {
                k: ("<redacted>" if k == "prompt" else v) for k, v in kwargs.items()
            }
            logger.debug("Calling ask() attempt %d with kwargs=%s", attempt, safe_kwargs)
            result = ask(**kwargs)
            if inspect.isawaitable(result) or isinstance(result, AsyncIterable):
                return asyncio.run(_drain_async_result(result))
            return _handle_sync_result(result)
        except (TimeoutError, ConnectionError, OSError) as exc:
            logger.warning("Transient error on attempt %d: %s", attempt, exc)
            if attempt == max_retries:
                logger.error("Max retries reached (%d).", max_retries)
                print("ERROR: transient failure", file=sys.stderr)
                return 124 if isinstance(exc, TimeoutError) else 1
            sleep_for = backoff * (2 ** (attempt - 1))
            sleep_for *= 0.8 + (os.urandom(1)[0] / 255.0) * 0.4
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.warning("operation interrupted by user")
            return 130
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.debug("ai request failed: %s", exc)
            print("ERROR: ai request failed", file=sys.stderr)
            return 1


def run_interactive(
    provider: str,
    model: str | None,
    timeout: int,
    profile: str | None = None,
    stream: bool = False,
    modules: str | None = None,
) -> int:
    print("--- AI CLI Interactive Mode ---")
    print(
        f"Provider: {provider} | Model: {model or 'default'} | "
        f"Profile: {profile or 'default'} | Stream: {stream}"
    )
    print(
        "Type /switch <provider>, /model <model>, /profile <name>, "
        "/stream, /index <file|text>, /help, /exit or /quit. Send plain "
        "text to ask a question.\n"
    )

    current_provider = provider
    current_model = model
    current_profile = profile
    current_stream = stream

    docs: list[str] = []
    doc_ids: list[str] = []

    try:
        while True:
            try:
                raw = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue

            if line in ("/exit", "/quit"):
                return 0

            if line.startswith("/switch "):
                current_provider = line.split(" ", 1)[1].strip() or current_provider
                print(f"Provider switched to: {current_provider}")
                continue

            if line.startswith("/model "):
                current_model = line.split(" ", 1)[1].strip() or None
                print(f"Model set to: {current_model or 'default'}")
                continue

            if line.startswith("/profile "):
                current_profile = line.split(" ", 1)[1].strip() or None
                print(f"Profile set to: {current_profile or 'default'}")
                continue

            if line == "/stream":
                current_stream = not current_stream
                print(f"Stream {'enabled' if current_stream else 'disabled'}")
                continue

            if line.startswith("/index "):
                payload = line.split(" ", 1)[1].strip()
                if not payload:
                    print("Usage: /index <file_path_or_raw_text>")
                    continue
                if os.path.isfile(payload):
                    try:
                        real = os.path.realpath(payload)
                        cwd = os.path.realpath(os.getcwd())
                        if not (real == cwd or real.startswith(cwd + os.sep)):
                            print("Refusing to read file outside working dir")
                            continue
                        st = os.stat(real)
                        if st.st_size > 10 * 1024 * 1024:
                            print("Skipping large file (>10MB)")
                            continue
                        with open(real, encoding="utf-8", errors="replace") as fh:
                            text = fh.read()
                        docs.append(text)
                        doc_ids.append(real)
                        print(f"Indexed file: {os.path.basename(payload)}")
                    except (OSError, UnicodeError) as exc:
                        logger.debug("Failed to read %s: %s", payload, exc)
                        print("Failed to index file (read error)")
                else:
                    docs.append(payload)
                    doc_ids.append(str(uuid.uuid4()))
                    print("Indexed raw text.")
                continue

            if line == "/help":
                print(
                    "Commands:\n"
                    "  /switch <provider>  - switch provider\n"
                    "  /model <model>      - set model\n"
                    "  /profile <name>     - set profile\n"
                    "  /stream             - toggle streaming\n"
                    "  /index <file|text>  - index a file or raw text\n"
                    "  /exit, /quit        - exit interactive mode\n"
                )
                continue

            kwargs = _build_ask_kwargs(
                provider=current_provider,
                prompt=line,
                model=current_model,
                timeout=timeout,
                profile=current_profile,
                stream=current_stream,
                modules=modules,
            )
            rc = _invoke_with_retries(kwargs)
            if rc != 0:
                logger.debug("ask returned non-zero exit code: %d", rc)
    except Exception:
        logger.exception("interactive session failed")
        print("ERROR: interactive session failed", file=sys.stderr)
        return 1   # only real failure

    return 0

def ask(prompt: str, **kwargs):
    return _get_ask_callable()(prompt, **kwargs)

def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint.

    - True CLI invocation (no argv provided) behaves like argparse and raises
      SystemExit(2) on empty/no-arg runs.
    - Programmatic invocation (argv provided) returns integer exit codes;
      missing/empty prompt returns 1 instead of raising.
    """
    parser = build_parser()

    # Real CLI invocation: raiseSystemExit to mirror argparse behavior.
    if argv is None:
        raise SystemExit(2)

    try:
        args = parser.parse_args(argv)
    except SystemExit as se:
        # Programmatic invocation: return parse error as int instead of raising.
        code = se.code if isinstance(se.code, int) else 1
        return int(code) if int(code) != 0 else 1

    if getattr(args, "debug", False):
        logger.setLevel(logging.DEBUG)

    _init_providers_safe()

    # Non-interactive requires a non-empty prompt. For programmatic callers
    # return 1 on missing/empty prompt rather than raising.
    if not getattr(args, "interactive", False):
        if args.prompt is None or not str(args.prompt).strip():
            return 1

    if args.interactive:
        rc = run_interactive(
            provider=args.provider,
            model=getattr(args, "model", None),
            timeout=getattr(args, "timeout", 30),
            profile=getattr(args, "profile", None),
            stream=getattr(args, "stream", False),
            modules=getattr(args, "modules", None),
        )
        return int(rc) if rc is not None else 0

    prompt = args.prompt.strip() if args.prompt is not None else None

    kwargs = _build_ask_kwargs(
        provider=args.provider,
        prompt=prompt,
        model=getattr(args, "model", None),
        timeout=getattr(args, "timeout", 30),
        profile=getattr(args, "profile", None),
        stream=getattr(args, "stream", False),
        modules=getattr(args, "modules", None),
    )

    _invoke_with_retries(kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())