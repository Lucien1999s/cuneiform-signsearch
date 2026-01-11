from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image

from src.embedding import Embedder, EmbedderConfig
from src.preprocess import PreprocessConfig, iter_views


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 5
    data_dir: Path = Path("data")


def _load_index(data_dir: Path):
    import faiss

    index_path = data_dir / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError("Index not found. Run `python3 src/indexer.py` first.")
    return faiss.read_index(str(index_path))


def _load_metadata(data_dir: Path) -> list[dict]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Metadata not found. Run `python3 src/indexer.py` first.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _merge_results(
    all_scores: dict[int, float],
    scores: np.ndarray,
    indices: np.ndarray,
) -> None:
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        current = all_scores.get(int(idx))
        if current is None or score > current:
            all_scores[int(idx)] = float(score)


def search_signs(
    image: Image.Image,
    embedder: Embedder,
    preprocess: PreprocessConfig,
    config: SearchConfig,
) -> list[dict]:
    index = _load_index(config.data_dir)
    metadata = _load_metadata(config.data_dir)

    all_scores: dict[int, float] = {}
    for view in iter_views(image, preprocess):
        embedding = embedder.embed([view])
        scores, indices = index.search(embedding, config.top_k)
        _merge_results(all_scores, scores, indices)

    ranked = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)[: config.top_k]
    results = []
    for idx, score in ranked:
        entry = metadata[idx]
        results.append(
            {
                "sign_id": entry.get("sign_id"),
                "english": entry.get("english"),
                "image_path": entry.get("image_path"),
                "score": score,
            }
        )
    return results


def default_embedder() -> Embedder:
    return Embedder(EmbedderConfig())
