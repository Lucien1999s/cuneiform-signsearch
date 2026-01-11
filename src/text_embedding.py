from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TextEmbedConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    model_dir: Path = Path("models") / "hf"


def _ensure_model_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(model_dir)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_texts(texts: Iterable[str], config: TextEmbedConfig) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    _ensure_model_dir(config.model_dir)
    model = SentenceTransformer(config.model_name, cache_folder=str(config.model_dir))

    prepared = [f"passage: {text}" for text in texts]
    vectors = model.encode(
        prepared,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return _normalize_vectors(vectors.astype(np.float32))


def embed_query(query: str, config: TextEmbedConfig) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    _ensure_model_dir(config.model_dir)
    model = SentenceTransformer(config.model_name, cache_folder=str(config.model_dir))

    prepared = f"query: {query}"
    vector = model.encode(prepared, convert_to_numpy=True, normalize_embeddings=False)
    vector = vector.reshape(1, -1).astype(np.float32)
    return _normalize_vectors(vector)
