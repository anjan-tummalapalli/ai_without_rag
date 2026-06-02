# /Users/anjan/Documents/New project/ai_chat/ai_cli/src/ai_cli/rag/embeddings.py
from __future__ import annotations
from typing import Iterable, List, Optional, TYPE_CHECKING, Any
from ai_cli.config.rag_config import EMBEDDING_MODEL
import os
# Import sentence_transformers at runtime; keep the TYPE_CHECKING import for static type checkers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import numpy as np
    from numpy.typing import NDArray  # type: ignore
except ImportError:
    np = None  # type: ignore
    NDArray = Any  # type: ignore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore

TESTING = os.getenv("PYTEST_RUNNING") == "1"


class EmbeddingGenerator:
    """
    Class that wraps a SentenceTransformers model to produce vector embeddings for text.
    Provides batching, optional L2-normalization, and convenience methods for single and
    batch encoding.

    Attributes:
        np: Reference to the numpy module used for numerical operations.
        batch_size: Number of texts to encode per call to the underlying model.
        normalize: If True, produced embeddings are L2-normalized.
        model: Instantiated SentenceTransformer model used for encoding.
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
        
    def _postprocess(self, emb: Any) -> Any:
        if not self.normalize:
            return emb
        if np is None:
            raise ImportError("numpy is required for normalization")
        arr = np.asarray(emb)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
    
    def embed_batch(self, texts: Iterable[str]) -> List[Any]:
        """
        Generate embeddings for an iterable of texts. Returns list of numpy arrays.
        Uses batching to avoid OOM on large lists.
        """
        texts = list(texts)
        embeddings: List[Any] = []
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