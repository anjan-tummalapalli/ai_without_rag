# Usage Instructions for AI CLI

## Overview

The AI CLI is a command-line interface tool that allows users to interact with various AI providers. It supports multiple AI models and provides a flexible way to send prompts and receive responses.

## Installation

To install the AI CLI, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd ai_cli
pip install -r requirements.txt
```

## Usage

### Running the CLI

You can run the AI CLI directly from the command line. The basic syntax is:

```bash
python -m ai_cli <command> [options]
```

### Commands

1. **Single-shot Prompt**

   To send a single prompt to a specific AI provider, use the following command:

   ```bash
   python -m ai_cli --provider <provider> --prompt "<your_prompt>"
   ```

   **Example:**

   ```bash
   python -m ai_cli --provider openai --prompt "What is the capital of France?"
   ```

   **Expected Output:**

   ```
   The capital of France is Paris.
   ```

2. **Interactive Session**

   To start an interactive chat session, use the `--interactive` flag:

   ```bash
   python -m ai_cli --interactive --provider <provider>
   ```

   **Example:**

   ```bash
   python -m ai_cli --interactive --provider claude
   ```

   You can then type your prompts directly into the terminal.

3. **List Available Models**

   To list all available models for a specific provider, use the `--list-models` flag:

   ```bash
   python -m ai_cli --provider <provider> --list-models
   ```

   **Example:**

   ```bash
   python -m ai_cli --provider gemini --list-models
   ```

4. **Piping Input**

   You can also pipe input from a file or another command:

   ```bash
   echo "Summarize this" | python -m ai_cli --provider gemini
   ```

   or

   ```bash
   cat notes.txt | python -m ai_cli --provider claude
   ```

## Environment Variables

Make sure to set the required environment variables for the AI providers you intend to use. Refer to the `.env.example` file for guidance on the necessary keys.

## Help Command

For more information on available commands and options, you can always use the `--help` flag:

```bash
python -m ai_cli --help
```

This will display a list of commands and their descriptions.