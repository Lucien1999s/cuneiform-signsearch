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
from src.mapping import cuneiform_icon_paths, cuneiform_translated_words
from src.preprocess import PreprocessConfig, normalize_image


@dataclass(frozen=True)
class SignEntry:
    sign_id: str
    image_path: str
    english: str


def _iter_entries(assets_dir: Path) -> Iterable[SignEntry]:
    if len(cuneiform_icon_paths) != len(cuneiform_translated_words):
        raise ValueError(
            "cuneiform_icon_paths and cuneiform_translated_words length mismatch: "
            f"{len(cuneiform_icon_paths)} vs {len(cuneiform_translated_words)}"
        )

    for icon, english in zip(cuneiform_icon_paths, cuneiform_translated_words):
        image_path = assets_dir / f"{icon}.jpg"
        if not image_path.exists():
            continue
        yield SignEntry(sign_id=icon, image_path=str(image_path), english=english)


def build_index(
    assets_dir: Path,
    output_dir: Path,
    embedder: Embedder,
    preprocess: PreprocessConfig,
    batch_size: int,
) -> None:
    import faiss

    entries = list(_iter_entries(assets_dir))
    if not entries:
        raise RuntimeError("No sign images found to index.")

    images: list[Image.Image] = []
    for entry in entries:
        image = Image.open(entry.image_path).convert("L")
        images.append(normalize_image(image, preprocess))

    embeddings = embedder.embed_batched(images, batch_size=batch_size)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    metadata = [entry.__dict__ for entry in entries]
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index for cuneiform signs.")
    parser.add_argument("--assets", default="assets", help="Path to sign images folder.")
    parser.add_argument("--output", default="data", help="Output folder for index files.")
    parser.add_argument("--model", default=EmbedderConfig().model_name, help="Model name.")
    parser.add_argument("--device", default="cpu", help="Device, e.g. cpu or cuda.")
    parser.add_argument("--weights", default=None, help="Optional local weights path.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    args = parser.parse_args()

    assets_dir = Path(args.assets)
    output_dir = Path(args.output)
    embedder = Embedder(
        EmbedderConfig(model_name=args.model, device=args.device, weights_path=args.weights)
    )
    preprocess = PreprocessConfig()
    build_index(assets_dir, output_dir, embedder, preprocess, batch_size=args.batch_size)
    print(f"Index built with assets from {assets_dir} -> {output_dir}")


if __name__ == "__main__":
    main()
