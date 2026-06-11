import os

def resolve_api_key(provider: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env_key = f"{provider.upper()}_API_KEY"
    return os.getenv(env_key)
