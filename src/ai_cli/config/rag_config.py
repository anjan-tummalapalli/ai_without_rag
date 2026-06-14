import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Legacy compatibility constants
# ------------------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.json"
VECTOR_STORE_TYPE = "faiss"
SIMILARITY_METRIC = "cosine"
RECREATE_INDEX = False


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


@dataclass
class RAGConfig:
    base_dir: Path
    index_dir: Path
    docs_dir: Path
    metadata_dir: Path
    metadata_path: Path

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    chunk_strategy: str = "recursive"
    top_k: int = TOP_K

    embedding: EmbeddingConfig = field(
        default_factory=EmbeddingConfig
    )

    vector_store: VectorStoreConfig = field(
        default_factory=VectorStoreConfig
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if not (0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError(
                "0 <= chunk_overlap < chunk_size must hold"
            )

        if self.top_k <= 0:
            self.top_k = 1

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, prefix: str = "RAG_") -> "RAGConfig":
        """Create a RAGConfig instance using environment variables with sensible defaults.

        The method constructs a base configuration with default paths and then overrides
        numeric parameters from the environment if they are set.
        """
        # Default configuration with placeholder paths; callers can adjust directories later.
        cfg = cls(
            base_dir=Path("."),
            index_dir=Path("./index"),
            docs_dir=Path("./docs"),
            metadata_dir=Path("./metadata"),
            metadata_path=Path("./metadata/metadata.json"),
        )
        # Override numeric and other simple fields from env variables.
        cfg.chunk_size = int(os.getenv(prefix + "CHUNK_SIZE", CHUNK_SIZE))
        cfg.chunk_overlap = int(os.getenv(prefix + "CHUNK_OVERLAP", CHUNK_OVERLAP))
        cfg.top_k = int(os.getenv(prefix + "TOP_K", TOP_K))
        return cfg

    def to_dict(self) -> dict:
        data = asdict(self)

        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        return convert(data)

    @classmethod
    def from_json(cls, path: Path) -> "RAGConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for key in (
            "base_dir",
            "index_dir",
            "docs_dir",
            "metadata_dir",
            "metadata_path",
        ):
            if key in data:
                data[key] = Path(data[key])

        if "embedding" in data:
            data["embedding"] = EmbeddingConfig(
                **data["embedding"]
            )

        if "vector_store" in data:
            data["vector_store"] = VectorStoreConfig(
                **data["vector_store"]
            )

        return cls(**data)