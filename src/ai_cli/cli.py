from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from ai_cli.ai_chat import ask

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
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
            "If omitted, stdin is used."
        ),
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help=(
            "Optional model override "
            "for the selected provider."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help=(
            "Request timeout in seconds. "
            "Default: 60"
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Main CLI entrypoint.

    Returns:
        int: process exit code
    """

    parser = build_parser()

    args = parser.parse_args(argv)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    prompt = args.prompt

    # Support stdin piping
    if not prompt:
        prompt = sys.stdin.read().strip()

    if not prompt:
        parser.error(
            "Prompt is required via "
            "--prompt or stdin."
        )

    logger.info(
        "provider=%s model=%s",
        args.provider,
        args.model,
    )

    try:

        response = ask(
            provider=args.provider,
            prompt=prompt,
            model=args.model,
            timeout=args.timeout,
        )

        print(response)

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "operation interrupted by user"
        )

        return 130

    except Exception as exc:

        logger.exception(
            "ai request failed: %s",
            exc,
        )

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())