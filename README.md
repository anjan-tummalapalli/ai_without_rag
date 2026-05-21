# AI CLI Chat Tool

## Overview

The AI CLI Chat Tool is a command-line interface application that allows users to interact with various AI providers. It supports multiple models and provides a flexible way to send prompts and receive responses from different AI systems.

## Features

- Supports multiple AI providers including OpenAI, Google Gemini, Anthropic Claude, and more.
- Interactive command-line interface for real-time chat.
- Ability to list available models for each provider.
- Environment variable management for API keys.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

You can run the AI CLI Chat Tool directly from the command line. Here are some examples:

### Single-shot prompt

```bash
python -m ai_cli -p openai -q "What is Python?"
```

### Interactive session

```bash
python -m ai_cli --interactive --provider claude
```

### List available models

```bash
python -m ai_cli --list-models
```

## Environment Variables

The following environment variables are required for the application to function properly. You can set them in a `.env` file or directly in your environment:

- `OPENAI_API_KEY` — OpenAI ChatGPT
- `GEMINI_API_KEY` — Google Gemini
- `ANTHROPIC_API_KEY` — Anthropic Claude
- `PERPLEXITY_API_KEY` — Perplexity AI
- `XAI_API_KEY` — xAI Grok
- `MISTRAL_API_KEY` — Mistral AI
- `GROQ_API_KEY` — Groq
- `DEEPSEEK_API_KEY` — DeepSeek
- `TOGETHER_API_KEY` — Together AI
- `COHERE_API_KEY` — Cohere
- `OPENROUTER_API_KEY` — OpenRouter
- `FIREWORKS_API_KEY` — Fireworks AI
- `GITHUB_TOKEN` — GitHub Models (Copilot API)

## Contributing

Contributions are welcome! Please refer to the [DEVELOPMENT.md](DEVELOPMENT.md) file for guidelines on how to set up your development environment and contribute to the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.