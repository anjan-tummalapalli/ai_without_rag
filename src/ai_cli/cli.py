from __future__ import annotations
import argparse, logging, sys
from typing import Any, Sequence
from ai_cli.ai_chat import ask

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
VERSION = "0.1.0"

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
    """
    Build and return the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="ai-cli",
        description=(
            "Enterprise AI CLI Gateway "
            "for multi-provider AI interactions."
        ),
    )

    parser.add_argument(
        "-p",
        "--provider",
        required=True,
        type=str,
        help=(
            "AI provider name "
            "(e.g. openai, gemini, claude, grok, cohere)."
        ),
    )

    parser.add_argument(
        "-q",
        "--prompt",
        type=str,
        help=(
            "Prompt/question to send to the AI provider. "
            "If omitted and stdin is piped, stdin is used."
        ),
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help=("Optional model override " "for the selected provider."),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help=("Request timeout in seconds. " "Default: 60"),
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
# Main Execution
# -----------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """
    Main CLI entrypoint for the ai_cli application.

    This function parses command-line arguments, reads a prompt either from the
    --prompt argument or from stdin (when piped), invokes the AI request via
    ask(...), and prints the response to stdout. It also handles logging,
    timeout validation, decoding of byte responses, and maps common failure
    scenarios to conventional exit codes.

    Args:
        argv (Sequence[str] | None): Optional list of command-line arguments to
            parse. If None, the parser reads arguments from sys.argv (default
            behaviour of argparse). Expected to be a sequence of strings or None.

    Returns:
        int: Process exit code indicating the outcome:
            - 0   : success, response printed to stdout
            - 1   : general error (parsing/IO/AI request failure)
            - 124 : request timed out
            - 130 : interrupted by user (KeyboardInterrupt)

    Side effects:
        - Reads from sys.stdin when no --prompt is provided and stdin is piped.
        - Writes the AI response to stdout.
        - Writes error messages to stderr on failure.
        - Adjusts logging level when --debug is set.
        - May call parser.error(...) which raises SystemExit for invalid args.

    Raises:
        SystemExit: If argument parsing fails (argparse parser.error is invoked).
        KeyboardInterrupt: Propagated/handled as an interrupted operation (returns 130).
        TimeoutError: Treated as a timeout condition (returns 124).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    prompt = args.prompt

    # Read from stdin only when piped (avoid blocking in interactive terminals)
    if not prompt:
        try:
            if not sys.stdin.isatty():
                prompt = sys.stdin.read().strip()
            else:
                # interactive terminal and no prompt provided -> error
                parser.error("Prompt is required via --prompt or stdin (piped).")
        except Exception as exc:
            logger.exception("failed to read stdin: %s", exc)
            print(f"ERROR: failed to read stdin: {exc}", file=sys.stderr)
            return 1

    if not prompt:
        parser.error("Prompt is required via " "--prompt or stdin.")

    if args.timeout is None or args.timeout <= 0:
        parser.error("timeout must be a positive integer")

    logger.info("provider=%s model=%s", args.provider, args.model)

    try:
        response = ask(
            provider=args.provider,
            prompt=prompt,
            timeout=args.timeout,
        )

        # Normalize response to string for safe printing
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
        # 124 is commonly used for timeout
        return 124

    except Exception as exc:
        logger.exception("ai request failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())