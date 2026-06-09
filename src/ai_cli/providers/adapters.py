class LegacyAskAdapter:

    def ask(self, prompt: str, **kwargs):

        return self.chat(prompt, **kwargs)