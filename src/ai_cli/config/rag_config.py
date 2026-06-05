# Add these imports near the top
import os
import importlib
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Callable, List, Dict
from pathlib import Path

# module logger
logger = logging.getLogger(__name__)

# Small dataclasses for embedding and vector store used by RAGConfig when constructing nested configs.
# These provide the minimal fields referenced in from_env and from_json so the names are defined.
@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 1
    normalize: bool = False

@dataclass
class VectorStoreConfig:
    store_type: str = "faiss"
    faiss_index_path: Path = Path("index.faiss")
    recreate: bool = False
    metric: str = "cosine"

# Tokenizer helper: prefer tiktoken when available
import re

def simple_tokenize(text: str) -> List[str]:
        """
        Simple fallback tokenizer: split text into word and punctuation tokens using a regex.
        Returns a list of string tokens.
        """
        # \w+ matches word characters (letters, digits, underscore), [^\s\w] matches any single
        # non-whitespace non-word character (punctuation). This keeps punctuation separate.
        return re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)

def get_tokenizer(prefer: Optional[str] = None) -> Callable[[str], List[str]]:
        """
        Return a tokenizer function. If 'tiktoken' is available and prefer is 'tiktoken',
        use it (requires tiktoken package). Otherwise fall back to simple_tokenize.
        """
        if prefer == "tiktoken":
                try:
                        tiktoken = importlib.import_module("tiktoken")
                        enc = tiktoken.get_encoding("cl100k_base")

                        def tok(text: str) -> List[str]:
                                # encode -> tokens are ints; map to strings for chunking semantics
                                return [str(x) for x in enc.encode(text)]
                        return tok
                except Exception:
                        logger.debug("tiktoken not available, falling back to simple_tokenize")
        return simple_tokenize


# Update/extend RAGConfig with post-init validation and env loading
@dataclass
class RAGConfig:
        # ... existing fields unchanged ...

        def __post_init__(self) -> None:
                # Ensure directories and validate sensible numeric params.
                try:
                        self.ensure_dirs()
                except Exception as e:
                        logger.warning("Could not create configured directories: %s", e)
                # normalize numeric settings
                if self.chunk_size <= 0:
                        raise ValueError("chunk_size must be > 0")
                if not (0 <= self.chunk_overlap < self.chunk_size):
                        raise ValueError("0 <= chunk_overlap < chunk_size must hold")
                if self.top_k <= 0:
                        logger.info("top_k <= 0, defaulting to 1")
                        self.top_k = 1

        @classmethod
        def from_env(cls, prefix: str = "RAG_") -> "RAGConfig":
                """
                Create config with overrides from environment variables.
                Supported keys: RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K,
                RAG_STORE_TYPE, RAG_FAISS_INDEX_PATH, RAG_EMBEDDING_MODEL
                """
                def getenv_int(key: str, default: int) -> int:
                        v = os.getenv(prefix + key)
                        return int(v) if v and v.isdigit() else default

                # Default configuration instance to pull defaults from, especially for nested configs. 
                # This ensures that if we add new fields
                default_config = RAGConfig(
                                           base_dir=Path("."),
                                           index_dir=Path("./index"),
                                           docs_dir=Path("./docs"),
                                           metadata_dir=Path("./metadata"),
                                           metadata_path=Path("./metadata/metadata.json"),
                                           chunk_size=500,
                                           chunk_overlap=50,
                                           chunk_strategy="recursive",
                                           top_k=5,
                                           embedding=EmbeddingConfig(),
                                           vector_store=VectorStoreConfig(),
                                          )

                                # ------------------------------------------------------------------
                                # Backward compatibility exports
                                # ------------------------------------------------------------------

                CHUNK_SIZE = default_config.chunk_size
                CHUNK_OVERLAP = default_config.chunk_overlap
                TOP_K = default_config.top_k
                
                # start from default and override simple fields
                cfg = default_config

                chunk_size = getenv_int("CHUNK_SIZE", cfg.chunk_size)
                chunk_overlap = getenv_int("CHUNK_OVERLAP", cfg.chunk_overlap)
                top_k = getenv_int("TOP_K", cfg.top_k)

                store_type = os.getenv(prefix + "STORE_TYPE", cfg.vector_store.store_type)
                faiss_path = os.getenv(prefix + "FAISS_INDEX_PATH", str(cfg.vector_store.faiss_index_path))

                emb_model = os.getenv(prefix + "EMBEDDING_MODEL", cfg.embedding.model_name)

                # build new nested configs
                emb = EmbeddingConfig(model_name=emb_model, batch_size=cfg.embedding.batch_size, normalize=cfg.embedding.normalize)
                vs = VectorStoreConfig(store_type=store_type, faiss_index_path=Path(faiss_path), recreate=cfg.vector_store.recreate, metric=cfg.vector_store.metric)

                return cls(
                        base_dir=cfg.base_dir,
                        index_dir=cfg.index_dir,
                        docs_dir=cfg.docs_dir,
                        metadata_dir=cfg.metadata_dir,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_strategy=cfg.chunk_strategy,
                        top_k=top_k,
                        embedding=emb,
                        vector_store=vs,
                        metadata_path=cfg.metadata_path,
                )

        def to_dict(self) -> Dict:
                """Return serializable dict representation. Converts Path objects to strings."""
                d = asdict(self)
                # asdict already handles nested dataclasses, but convert Path -> str recursively
                def convert(obj):
                        if isinstance(obj, dict):
                                return {k: convert(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                                return [convert(v) for v in obj]
                        if isinstance(obj, str):
                                return obj
                        if isinstance(obj, Path):
                                return str(obj)
                        return obj
                return convert(d)

        @classmethod
        def from_json(cls, path: Path) -> "RAGConfig":
                with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                # coerce path strings
                for k in ("base_dir", "index_dir", "docs_dir", "metadata_dir", "metadata_path"):
                        if k in data and data[k]:
                                data[k] = Path(data[k])
                emb = EmbeddingConfig(**data.get("embedding", {}))
                vs = VectorStoreConfig(**data.get("vector_store", {}))
                kwargs = {k: data[k] for k in data.keys() if k not in {"embedding", "vector_store"}}
                return cls(**kwargs, embedding=emb, vector_store=vs)
