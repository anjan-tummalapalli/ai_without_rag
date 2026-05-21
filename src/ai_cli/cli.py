import argparse
from ai_cli.ai_chat import ask

def main():
    parser = argparse.ArgumentParser(
        description="AI CLI Chat Tool - Interact with various AI providers."
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        required=True,
        help="Specify the AI provider to use (e.g., openai, gemini, claude)."
    )
    parser.add_argument(
        "--prompt",
        "-q",
        type=str,
        required=True,
        help="The prompt or question to send to the AI."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Optional model name to override the default for the provider."
    )
    args = parser.parse_args()

    response = ask(args.provider, args.prompt, args.model)
    print(response)

if __name__ == "__main__":
    main()