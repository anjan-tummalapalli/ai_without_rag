from ..contracts import ChatProvider, EmbeddingProvider
from ..base import BaseProvider, ProviderConfig
from ..registry import register_provider

@register_provider("deepseek")
class DeepSeekProvider(BaseProvider, ChatProvider, EmbeddingProvider):
    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_EMBED = "text-embedding-3-small"

    def __init__(self, config: ProviderConfig | None = None, **kwargs):
        config = config or ProviderConfig(**kwargs)
        super().__init__(config)
        self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY missing")
        self.client = OpenAI(
                               api_key=self.api_key,
                               base_url=self.BASE_URL,
                            )

        self.embed_model = config.embedding_model or self.DEFAULT_EMBED
    # ---------------- CHAT ----------------

    def chat(self, prompt: str, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model or self.DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return resp.choices[0].message.content.strip()

    # -------------- EMBED -----------------

    def embed(self, texts: list[str], **kwargs):
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=texts
        )
        return [d.embedding for d in resp.data]
