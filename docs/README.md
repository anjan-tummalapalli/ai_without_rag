# AI CLI Chat Tool

## Overview

The AI CLI Chat Tool is a command-line interface application that allows users to interact with various AI providers. It supports multiple models and provides a flexible way to send prompts and receive responses from different AI services.

## Features

- Supports multiple AI providers including OpenAI, Google Gemini, Anthropic Claude, and more.
- Interactive command-line interface for real-time chat.
- Ability to list available models for each provider.
- Environment variable management for API keys.

## Installation

To install the AI CLI Chat Tool, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/ai_cli.git
   cd ai_cli
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables. Create a `.env` file based on the `.env.example` provided in the repository.

## Usage

To start using the AI CLI Chat Tool, run the following command:

```
python -m ai_cli
```

You can also use specific commands to interact with the AI providers. For example:

```
python -m ai_cli --provider openai --prompt "What is Python?"
```

For detailed usage instructions, refer to the [USAGE.md](USAGE.md) file.

## Documentation

- [USAGE.md](USAGE.md): Detailed usage instructions and examples.
- [API.md](API.md): Documentation of the API, including available functions and parameters.
- [DEVELOPMENT.md](DEVELOPMENT.md): Guidelines for contributing to the project.

## Contributing

Contributions are welcome! Please read the [DEVELOPMENT.md](DEVELOPMENT.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.