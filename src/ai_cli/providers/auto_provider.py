from __future__ import annotations
import os
import logging
from typing import List, Tuple, Optional, Any, Dict

try:
    import numpy as np
except Exception:
    np = None

# optional dependencies
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None

try:
    import faiss
except Exception:
    faiss = None

logger = logging.getLogger("ai_gateway.rag")


def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required; install it with 'pip install numpy'")


class EmbeddingBackendUnavailable(RuntimeError):
    pass


class AdvancedRAG:
    """
    Advanced RAG helpers: chunking, embedding, FAISS vector store creation and querying.

    Design goals:
    - Chunk arbitrary texts into overlapping passages.
    - Produce embeddings using either OpenAI or SentenceTransformers.
    - Build an in-memory FAISS index and simple metadata store.
    - Query the index by embedding the query and returning nearest passages with scores.

    Usage example:
      rag = AdvancedRAG(backend_preference=["openai", "sbert"])
      docs = ["Long document text ...", "Another doc ..."]
      chunks, meta = rag.chunk_texts(docs, chunk_size=1000, overlap=200)
      embeddings = rag.embed_texts(chunks)
      index, store = rag.build_faiss_index(embeddings, meta)
      results = rag.query_index(index, store, "What is the ...?", top_k=5)
    """

    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2",
        backend_preference: List[str] = ["openai", "sbert"],
    ) -> None:
        self.sbert_model = sbert_model
        self.backend_preference = backend_preference
        self._sbert: Optional[Any] = None
        self._openai_key = os.environ.get("OPENAI_API_KEY")
        if openai is not None and self._openai_key:
            openai.api_key = self._openai_key

    # ---- chunking ----
    def chunk_texts(
        self,
        texts: List[str],
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Chunk list of input texts into overlapping chunks.
        Returns: (chunks, metas) where metas contains {source_doc_index, chunk_index, start, end}
        """
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        for doc_idx, text in enumerate(texts):
            if not text:
                continue
            length = len(text)
            if length <= chunk_size:
                chunks.append(text)
                metas.append({"source_doc": doc_idx, "chunk_index": 0, "start": 0, "end": length})
                continue
            step = chunk_size - overlap
            if step <= 0:
                raise ValueError("chunk_size must be larger than overlap")
            start = 0
            chunk_idx = 0
            while start < length:
                end = min(start + chunk_size, length)
                chunk = text[start:end]
                chunks.append(chunk)
                metas.append({"source_doc": doc_idx, "chunk_index": chunk_idx, "start": start, "end": end})
                chunk_idx += 1
                if end == length:
                    break
                start += step
        return chunks, metas

    def _init_sbert(self) -> None:
        """
        Lazily initialize the SentenceTransformer model.
        """
        if self._sbert is None:
            if SentenceTransformer is None:
                raise EmbeddingBackendUnavailable("SentenceTransformers is not installed")
            self._sbert = SentenceTransformer(self.sbert_model)

    def _embed_with_openai(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Use OpenAI embeddings API (text-embedding-3-small or text-embedding-3-large).
        """
        _require_numpy()
        if openai is None or not self._openai_key:
            raise EmbeddingBackendUnavailable("OpenAI not configured")
        model = "text-embedding-3-small"
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = openai.Embeddings.create(model=model, input=batch)
            for item in resp["data"]:
                vec = np.asarray(item["embedding"], dtype=np.float32)
                embs.append(vec)
        return embs

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Embed a list of texts. Selects backend according to preference and availability.
        Returns list of 1D numpy arrays (float32).
        """
        _require_numpy()
        backends = [b.lower() for b in self.backend_preference]
        for b in backends:
            if b == "openai":
                if openai is not None and self._openai_key:
                    try:
                        return self._embed_with_openai(texts, batch_size=batch_size)
                    except Exception as exc:
                        logger.warning("openai embeddings failed: %s", exc)
                        continue
            if b in ("sbert", "sentence-transformers", "sentence_transformers"):
                if SentenceTransformer is not None:
                    try:
                        self._init_sbert()
                        embs: List[np.ndarray] = []
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i : i + batch_size]
                            arr = self._sbert.encode(batch, convert_to_numpy=True)
                            # arr is a 2D numpy array
                            for vec in arr:
                                embs.append(np.asarray(vec, dtype=np.float32))
                        return embs
                    except Exception as exc:
                        logger.warning("sbert embeddings failed: %s", exc)
                        continue
        raise EmbeddingBackendUnavailable("No embedding backend available or all backends failed")

    def build_faiss_index(
        self,
        embeddings: List[np.ndarray],
        metas: List[Dict[str, Any]],
        use_index: Optional[Any] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Build an in-memory FAISS index from embeddings.
        Returns (index, metadata_store) where metadata_store aligns with embeddings order.
        """
        _require_numpy()
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        if not embeddings:
            raise ValueError("no embeddings provided")
        dim = int(embeddings[0].shape[0])
        xb = np.vstack([e.reshape(1, -1) for e in embeddings]).astype("float32")
        # simple index: IndexFlatIP (inner product) after L2-normalize => cosine similarity
        index = faiss.IndexFlatIP(dim)
        # normalize
        faiss.normalize_L2(xb)
        index.add(xb)
        # store metas as-is; user is responsible for persistence if needed
        return index, metas

    def query_index(
        self,
        index: Any,
        metas: List[Dict[str, Any]],
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Query the provided index with text query. Returns list of (meta, score) sorted by score desc.
        """
        _require_numpy()
        q_embs = self.embed_texts([query])
        q = q_embs[0].reshape(1, -1).astype("float32")
        if faiss is None:
            raise RuntimeError("faiss not installed")
        faiss.normalize_L2(q)
        D, I = index.search(q, top_k)
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(metas):
                continue
            results.append((metas[int(idx)], float(score)))
        return results
