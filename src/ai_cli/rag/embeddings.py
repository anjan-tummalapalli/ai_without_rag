from __future__ import annotations
from typing import Iterable, List, TYPE_CHECKING, Any, Optional, Union
from ai_cli.config.rag_config import EMBEDDING_MODEL
import os
import importlib
import inspect

try:
    import numpy as _np  # local alias to avoid name clash until set on self
except Exception:
    _np = None  # type: ignore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

TESTING = os.getenv("PYTEST_RUNNING") == "1"


class EmbeddingGenerator:
    """
    Wrapper around a SentenceTransformer (or a compatible embedding model) that:
    - supports passing either a model name or an already-instantiated model,
    - batches calls to avoid OOM,
    - normalizes embeddings (optional),
    - tolerates different return types from model.encode (list, numpy, torch.Tensor).

    New/updated behaviors:
    - Accepts model: Union[str, Any] to allow pre-instantiated models or custom wrappers.
    - Optional device parameter to move model to a specific device, when available.
    - More robust handling of encode() kwargs and return types.
    - Uses instance-held numpy reference (self.np) to avoid global state issues.
    """

    def __init__(
        self,
        model: Union[str, Any] = EMBEDDING_MODEL,
        batch_size: int = 32,
        normalize: bool = True,
        device: Optional[str] = None,
    ) -> None:
        # Allow passing an instantiated model or a model name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device

        # Import sentence-transformers lazily if needed
        SentenceTransformerCls = None
        if isinstance(model, str):
            try:
                st_mod = importlib.import_module("sentence_transformers")
                SentenceTransformerCls = getattr(st_mod, "SentenceTransformer")
            except Exception:
                raise ImportError(
                    "The 'sentence-transformers' package is required but not installed. "
                    "Install it with: pip install sentence-transformers"
                )

        # Import numpy lazily and store reference on instance
        try:
            import numpy as numpy  # type: ignore
        except Exception:
            raise ImportError(
                "The 'numpy' package is required but not installed. "
                "Install it with: pip install numpy"
            )
        self.np = numpy

        # instantiate or use provided model
        if isinstance(model, str):
            self.model: Any = SentenceTransformerCls(model)
        else:
            self.model = model

        # try to move model to requested device if supported
        if self.device is not None:
            # SentenceTransformer has .to(), custom models may also
            try:
                to_fn = getattr(self.model, "to", None)
                if callable(to_fn):
                    to_fn(self.device)
            except Exception:
                # best-effort: ignore if model doesn't support moving
                pass

    def _postprocess(self, emb: Any) -> Any:
        if not self.normalize:
            return emb

        np = self.np
        if np is None:
            raise ImportError("numpy is required for normalization")

        arr = np.asarray(emb)
        if arr.ndim == 1:
            norm = np.linalg.norm(arr)
            if norm == 0:
                norm = 1.0
            return arr / norm

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _ensure_numpy(self, emb: Any) -> Any:
        """
        Convert various embedding return types to numpy.ndarray.
        Accepts: numpy arrays, lists of floats, torch tensors.
        """
        np = self.np
        if np is None:
            raise ImportError("numpy is required to handle embeddings")

        # torch.Tensor handling without hard dependency
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore

        if torch is not None and hasattr(torch, "Tensor") and isinstance(emb, torch.Tensor):
            return emb.detach().cpu().numpy()
        # model.encode sometimes returns a list or nested lists
        return np.asarray(emb)

    def _encode_with_fallback(self, batch: List[str]) -> Any:
        """
        Call model.encode with a best-effort set of kwargs. If the model doesn't accept
        certain kwargs, fall back to calling without them and convert the result.
        """
        encode_kwargs = {}
        # commonly supported kwargs
        encode_kwargs["batch_size"] = self.batch_size
        encode_kwargs["show_progress_bar"] = False
        # prefer convert_to_numpy when available
        encode_kwargs["convert_to_numpy"] = True
        # some versions accept device
        if self.device is not None:
            encode_kwargs["device"] = self.device

        # attempt to pass supported kwargs only
        try:
            sig = inspect.signature(self.model.encode)
            supported = {
                k for k in encode_kwargs.keys() if k in sig.parameters
            }
            call_kwargs = {k: encode_kwargs[k] for k in supported}
            result = self.model.encode(batch, **call_kwargs)
        except Exception:
            # final fallback: plain call
            result = self.model.encode(batch)

        # ensure numpy array
        try:
            result_np = self._ensure_numpy(result)
        except Exception:
            # if conversion fails, just return as-is (downstream may handle)
            return result
        return result_np

    def embed_batch(self, texts: Iterable[str]) -> List[Any]:
        """
        Generate embeddings for an iterable of texts. Returns list of numpy arrays.
        Uses batching to avoid OOM on large lists.
        """
        # Normalize input items to strings (support objects with .text)
        texts_list = [
            t.text if hasattr(t, "text") else str(t) for t in texts
        ]
        embeddings: List[Any] = []
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i : i + self.batch_size]
            emb_batch = self._encode_with_fallback(batch)
            # emb_batch can be a single vector for single input; ensure 2D
            emb_batch = self._postprocess(emb_batch)
            # convert to list of vectors
            if getattr(emb_batch, "ndim", None) == 1:
                embeddings.append(emb_batch)
            else:
                embeddings.extend(list(emb_batch))
        return embeddings

    def embed_text(self, text: str) -> Any:
        """
        Generate an embedding for a single text string.
        Returns a single embedding vector (numpy array).
        """
        # Reuse batching/encoding machinery for consistent behavior
        emb_list = self.embed_batch([text])
        return emb_list[0]

    @classmethod
    def from_model(cls, model_instance: Any, batch_size: int = 32, normalize: bool = True, device: Optional[str] = None):
        """
        Alternate constructor when you already have an instantiated model object.
        """
        return cls(model=model_instance, batch_size=batch_size, normalize=normalize, device=device)


# Backward compatibility alias (legacy provider API)
EmbeddingsProvider = EmbeddingGenerator