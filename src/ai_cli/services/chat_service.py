class ChatService:

    def __init__(self, provider):
        self.provider = provider

    def ask(self, prompt):
        return self.provider.ask(prompt)