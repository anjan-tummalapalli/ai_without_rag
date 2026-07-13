def resolve_provider_name(name: str) -> str:
    name = (name or "").lower().strip()
    if name == "auto":
        return "openai"  # can later upgrade to smarter routing
    return name
