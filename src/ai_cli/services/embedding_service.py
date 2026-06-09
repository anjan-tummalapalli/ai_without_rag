class EmbeddingService:

    def __init__(self, provider):
        self.provider = provider

    def embed(self, texts):
        return self.provider.embed(texts)
