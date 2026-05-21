# API Documentation for AI CLI

## Overview

The AI CLI is a command-line interface tool that allows users to interact with various AI providers. This document outlines the available functions, their parameters, and return values.

## Available Functions

### 1. `ask(provider: str, prompt: str, model: Optional[str] = None) -> str`

- **Description**: Dispatches a prompt to the specified AI provider and returns the AI's response.
- **Parameters**:
  - `provider` (str): The key of the AI provider to use (e.g., "openai", "gemini").
  - `prompt` (str): The user's question or instruction.
  - `model` (Optional[str]): The model identifier to use. If not specified, the provider's default model is used.
- **Returns**: The AI's response as a string. If an error occurs, it returns a string starting with "[ERROR]".

### 2. `run_interactive(provider: str, model: Optional[str] = None) -> None`

- **Description**: Starts an interactive REPL-style chat session with the specified AI provider.
- **Parameters**:
  - `provider` (str): The initial provider key (must be one of the keys in `PROVIDERS`).
  - `model` (Optional[str]): The initial model override. If not specified, the provider's default model is used.
- **Returns**: None. The function runs indefinitely until the user exits.

### 3. `print_banner() -> None`

- **Description**: Prints the ASCII-art application banner to stdout, displaying the tool name and supported providers.
- **Returns**: None.

### 4. `c(text: str, color: str) -> str`

- **Description**: Wraps the provided text in ANSI color codes for terminal output.
- **Parameters**:
  - `text` (str): The string to colorize.
  - `color` (str): The color name (e.g., "cyan", "green").
- **Returns**: The original text wrapped in ANSI escape sequences.

## Error Handling

All functions are designed to handle errors gracefully. If an error occurs during execution, the functions will return a string that begins with "[ERROR]", providing information about the issue.

## Example Usage

```python
response = ask("openai", "What is the capital of France?")
print(response)  # Expected output: "The capital of France is Paris."
```

## Conclusion

This API documentation provides a comprehensive overview of the functions available in the AI CLI tool. For further details on usage and examples, please refer to the `USAGE.md` document.