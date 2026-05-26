from __future__ import annotations
import argparse, logging, sys
from typing import Any, Sequence
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
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="Show program's version number and exit.",
    )

    return parser


# -----------------------------------------------------------------------------
# Interactive REPL Loop
# -----------------------------------------------------------------------------
def run_interactive(provider: str, model: str | None, timeout: int) -> int:
    """Run an interactive chat session."""
    print(f"--- AI CLI Interactive Mode ---")
    print(f"Current Provider: {provider}")
    print(f"Type /switch <provider> to change provider.")
    print(f"Type /exit or /quit to exit.\n")

    current_provider = provider

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return 0

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            return 0

        if user_input.startswith("/switch "):
            _, new_p = user_input.split(maxsplit=1)
            current_provider = new_p.strip()
            print(f"Switched provider to: {current_provider}")
            continue

        print(f"[{current_provider}] Thinking...")
        try:
            response = ask(
                provider=current_provider,
                prompt=user_input,
                model=model,
                timeout=timeout,
            )
            print(f"\n{current_provider}: {response}")
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)

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

    if args.interactive:
        return run_interactive(args.provider, args.model, args.timeout)

    prompt = args.prompt

    # Read from stdin only when piped
    if not prompt:
        try:
            if not sys.stdin.isatty():
                prompt = sys.stdin.read().strip()
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

    if args.timeout is None or args.timeout <= 0:
        parser.error("timeout must be a positive integer")

    logger.info("provider=%s model=%s", args.provider, args.model)

    try:
        response = ask(
            provider=args.provider,
            prompt=prompt,
            model=args.model,
            timeout=args.timeout,
        )

        if isinstance(response, bytes):
            try:
                response_text = response.decode("utf-8")
            except Exception:
                response_text = response.decode("utf-8", errors="replace")
        else:
            response_text = str(response)

        print(response_text)
        return 0

    except KeyboardInterrupt:
        logger.warning("operation interrupted by user")
        return 130
    except TimeoutError as exc:
        logger.error("request timed out: %s", exc)
        print(f"ERROR: request timed out: {exc}", file=sys.stderr)
        return 124
    except Exception as exc:
        logger.exception("ai request failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
