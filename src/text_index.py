from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from pymilvus import MilvusClient

from src.mapping import ICON_PATHS, TRANSLATED_WORDS
from src.text_embedding import TextEmbedConfig, embed_texts


def _build_records() -> list[dict]:
    if len(ICON_PATHS) != len(TRANSLATED_WORDS):
        raise ValueError(
            "ICON_PATHS and TRANSLATED_WORDS length mismatch: "
            f"{len(ICON_PATHS)} vs {len(TRANSLATED_WORDS)}"
        )

    records = []
    for idx, (icon, text) in enumerate(zip(ICON_PATHS, TRANSLATED_WORDS)):
        records.append(
            {
                "id": idx,
                "icon_path": f"datasets/{icon}.jpg",
                "text": text,
            }
        )
    return records


def build_index(
    output_dir: Path,
    collection_name: str,
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _build_records()

    vectors = embed_texts(
        [record["text"] for record in records],
        TextEmbedConfig(model_name=model_name),
    )

    client = MilvusClient(uri=str(output_dir / "milvus.db"))
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vector_field_name="embedding",
        dimension=vectors.shape[1],
        metric_type="IP",
        auto_id=False,
        primary_field_name="id",
    )

    payload = []
    for record, vector in zip(records, vectors):
        payload.append(
            {
                "id": record["id"],
                "icon_path": record["icon_path"],
                "text": record["text"],
                "embedding": vector.tolist(),
            }
        )

    client.insert(collection_name=collection_name, data=payload)
    client.flush(collection_name)
    print(f"Milvus index ready: {output_dir / 'milvus.db'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Milvus text index.")
    parser.add_argument("--output", default="data", help="Output directory.")
    parser.add_argument("--collection", default="tablets", help="Collection name.")
    parser.add_argument(
        "--model",
        default=TextEmbedConfig().model_name,
        help="HF embedding model name.",
    )
    args = parser.parse_args()

    build_index(Path(args.output), args.collection, args.model)


if __name__ == "__main__":
    main()
