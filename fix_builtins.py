import re

with open("src/ai_cli/plugins/builtins.py") as f:
    content = f.read()

# Fix OpenAIProvider
content = content.replace('def _send_impl(self, prompt: str) -> str:', 'def send(self, prompt: str, **kwargs) -> str:')
content = re.sub(
    r'def __init__\(self, model: str \| None = None\) -> None:(.*?super\(\)\.__init__\(.*?\))',
    r'def __init__(self, model: str | None = None):\1\n        self.timeout = 60.0',
    content,
    flags=re.DOTALL | re.MULTILINE
)
content = re.sub(
    r'def __init__\(self, provider_name: str, model: str \| None = None\) -> None:(.*?super\(\)\.__init__\(.*?\))',
    r'def __init__(self, provider_name: str, model: str | None = None):\1\n        self.timeout = 60.0\n        self.provider_name = provider_name',
    content,
    flags=re.DOTALL | re.MULTILINE
)

# Remove metrics from OpenAIProvider
content = re.sub(
    r'\s*usage = getattr\(response, "usage", None\)\n\s*if usage:\n\s*self\.metrics\.total_prompt_tokens \+= getattr\(\n\s*usage, "prompt_tokens", 0\n\s*\)\n\s*self\.metrics\.total_completion_tokens \+= getattr\(\n\s*usage, "completion_tokens", 0\n\s*\)',
    '',
    content,
    flags=re.MULTILINE
)

# Remove metrics from OpenAICompatibleProvider
content = re.sub(
    r'\s*usage = getattr\(response, "usage", None\)\n\s*if usage:\n\s*self\.metrics\.total_prompt_tokens \+= getattr\(usage, "prompt_tokens", 0\)\n\s*self\.metrics\.total_completion_tokens \+= getattr\(usage, "completion_tokens", 0\)',
    '',
    content,
    flags=re.MULTILINE
)

# Remove metrics from CohereProvider
content = re.sub(
    r'\s*# Try to update token metrics if available \(best-effort\)\n\s*try:\n\s*token_count = getattr\(response, "token_count", None\) or \(\n\s*\(getattr\(response, "meta", \{\}\) or \{\}\)\.get\("token_count"\)\n\s*\)\n\s*if token_count is not None:\n\s*self\.metrics\.total_prompt_tokens \+= int\(token_count\)\n\s*except Exception as exc:\n\s*logger\.debug\("Token metric update failed: %s", exc, exc_info=True\)',
    '',
    content,
    flags=re.MULTILINE
)

with open("src/ai_cli/plugins/builtins.py", "w") as f:
    f.write(content)

