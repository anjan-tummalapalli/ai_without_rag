class LegacyAskAdapter:
    """Adapter exposing the legacy ask() interface."""

    def ask(self, prompt: str, **kwargs):
        return self.chat(prompt, **kwargs)

    def chat(self, prompt: str, **kwargs):
        """Implement chat in subclasses or wrap the new provider interface."""
        raise NotImplementedError("chat() must be implemented by subclasses")
