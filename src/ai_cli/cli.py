from __future__ import annotations
"""
ai_cli.cli - Command-line interface for the Enterprise AI CLI Gateway.

This module implements the CLI entrypoint for interacting with the core
ai_cli.ask() API. It provides:

- Argument parsing for provider, model, prompt, interactive mode, streaming,
    timeout, debug and profile selection.
- A compatibility layer that inspects ask()'s signature and builds a safe set
    of kwargs to remain compatible across different ask() implementations.
- Handling for synchronous results, awaitables, async iterables (streaming),
    and generator/iterable results. Streaming parts are printed to stdout as
    they arrive.
- An interactive REPL mode with commands to switch provider/model/profile,
    toggle streaming, and exit.
- A retry wrapper for transient errors with exponential backoff.
- Logging to stderr and user-friendly error messages.

End result: the CLI prints AI responses to stdout (either streamed chunks or a
final result), writes logs and errors to stderr, and exits with standard codes:
0 on success, 130 on user interrupt, 124 on timeout, and non-zero on other
failures.
"""
import argparse, asyncio, inspect, json, logging, sys, time
from typing import Any, Iterable, AsyncIterable, Sequence
from ai_cli.core.api import ask

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
VERSION = "0.2.0"

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
# CLI Argument Parsing
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
        """Build and return the CLI argument parser."""
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
        model = (
                model.strip() if isinstance(model, str) and model.strip() else None
        )
        profile = (
                profile.strip() if isinstance(profile, str) and profile.strip() else None
        )

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
                accepts_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
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
                if isinstance(value, Iterable) and not isinstance(
                        value, (str, bytes, dict)
                ):
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
                        return asyncio.run(
                                _drain_async_result(result, provider, stream)
                        )

                # If it's an async iterable object instance (rare)
                if isinstance(result, AsyncIterable):
                        return asyncio.run(
                                _drain_async_result(result, provider, stream)
                        )

                # Iterable streaming (but strings are iterable of chars -> avoid)
                if isinstance(result, Iterable) and not isinstance(
                        result, (str, bytes, dict)
                ):
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
                        print(
                                json.dumps(
                                        result, indent=2, ensure_ascii=False, default=str
                                )
                        )
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
def _invoke_with_retries(kwargs: dict[str, Any], max_retries: int = 3,
                                                 backoff: float = 0.5) -> int:
        """
        Invoke ask() with a small retry/backoff on transient errors.
        Handles sync and async responses transparently. Returns an exit code.
        """
        attempt = 0
        while True:
                try:
                        attempt += 1
                        # Prepare a safe-to-log copy of kwargs
                        safe_kwargs = {
                                k: ("<redacted>" if k == "prompt" else v)
                                for k, v in kwargs.items()
                        }
                        logger.debug(
                                "Calling ask() attempt %d with kwargs=%s",
                                attempt,
                                safe_kwargs,
                        )
                        result = ask(**kwargs)
                        # If the result is awaitable or async iterable, handle via
                        # asyncio
                        if inspect.isawaitable(result) or isinstance(result, AsyncIterable):
                                return asyncio.run(
                                        _drain_async_result(
                                                result,
                                                kwargs.get("provider", "unknown"),
                                                kwargs.get("stream", False),
                                        )
                                )
                        # Otherwise handle sync
                        return _handle_sync_result(
                                result,
                                kwargs.get("provider", "unknown"),
                                kwargs.get("stream", False),
                        )
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
) -> int:
        """Run an interactive chat session."""
        print("--- AI CLI Interactive Mode ---")
        print(
                f"Provider: {provider} | Model: {model or 'default'} | "
                f"Profile: {profile or 'default'} | Stream: {stream}"
        )
        print(
                "Type /switch <provider>, /model <model>, /profile <name>, "
                "/stream, /exit or /quit. Type /help for this text.\n"
        )

        current_provider = provider
        current_model = model
        current_profile = profile
        current_stream = stream

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
                        print("  /switch <provider>    Switch provider")
                        print("  /model <model>        Set model override")
                        print("  /profile <name>       Set profile")
                        print("  /stream               Toggle streaming mode")
                        print("  /exit, /quit          Exit")
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

                # Normal message
                print(f"[{current_provider}] Thinking...")
                kwargs = _build_ask_kwargs(
                        provider=current_provider,
                        prompt=user_input,
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

        if args.interactive:
                return run_interactive(
                        args.provider,
                        args.model,
                        args.timeout,
                        profile=args.profile,
                        stream=args.stream,
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

        logger.info(
                "provider=%s model=%s profile=%s stream=%s timeout=%s",
                args.provider,
                args.model,
                args.profile,
                args.stream,
                args.timeout,
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
