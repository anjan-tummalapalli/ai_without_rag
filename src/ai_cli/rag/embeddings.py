# /Users/anjan/Documents/New project/ai_chat/ai_cli/src/ai_cli/rag/embeddings.py
from __future__ import annotations
from typing import Iterable, List, Optional, TYPE_CHECKING
from ai_cli.config.rag_config import EMBEDDING_MODEL
import os

# Top-level imports: try to import numpy and sentence_transformers once so
# tools and linters see the imports at module scope instead of inside functions.
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

if TYPE_CHECKING:
    import numpy as np  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

TESTING = os.getenv("PYTEST_RUNNING") == "1"


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers.
    - batch_size: number of texts to encode in each call to the model
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "The 'sentence-transformers' package is required but not installed. "
                "Install it with: pip install sentence-transformers"
            )
        if np is None:
            raise ImportError(
                "The 'numpy' package is required but not installed. "
                "Install it with: pip install numpy"
            )
        self.np = np
        self.batch_size = batch_size
        self.normalize = normalize
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def _postprocess(self, emb: np.ndarray) -> np.ndarray:
        # Normalize vectors if requested (L2 normalization per vector)
        if not self.normalize:
            return emb
        # ensure we operate with numpy available
        arr = self.np.asarray(emb)
        norms = self.np.linalg.norm(arr, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        """
        emb = self.model.encode([text], convert_to_numpy=True)
        return self._postprocess(emb)[0]

    def embed_batch(self, texts: Iterable[str]) -> List[np.ndarray]:
        """
        Generate embeddings for an iterable of texts. Returns list of numpy arrays.
        Uses batching to avoid OOM on large lists.
        """
        texts = list(texts)
        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            emb_batch = self.model.encode(batch, convert_to_numpy=True)
            emb_batch = self._postprocess(emb_batch)
            embeddings.extend(list(emb_batch))
        return embeddings


# ------------------------------------------------------------------
# Backward compatibility alias (legacy provider API)
# ------------------------------------------------------------------
EmbeddingsProvider = EmbeddingGenerator