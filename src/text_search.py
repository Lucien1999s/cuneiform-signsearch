from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from pymilvus import MilvusClient

from src.text_embedding import TextEmbedConfig, embed_query


def search(
    query: str,
    output_dir: Path,
    collection_name: str,
    top_k: int,
    model_name: str,
) -> list[dict]:
    vector = embed_query(query, TextEmbedConfig(model_name=model_name))
    client = MilvusClient(uri=str(output_dir / "milvus.db"))

    results = client.search(
        collection_name=collection_name,
        data=vector.tolist(),
        limit=top_k,
        output_fields=["icon_path", "text"],
        search_params={"metric_type": "IP"},
    )

    hits = []
    for hit in results[0]:
        score = hit.get("score")
        if score is None:
            score = hit.get("distance")
        hits.append(
            {
                "id": hit.get("id"),
                "score": score,
                "icon_path": hit.get("entity", {}).get("icon_path"),
                "text": hit.get("entity", {}).get("text"),
            }
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Milvus text index.")
    parser.add_argument("query", help="English query string.")
    parser.add_argument("--output", default="data", help="Output directory.")
    parser.add_argument("--collection", default="tablets", help="Collection name.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results.")
    parser.add_argument(
        "--model",
        default=TextEmbedConfig().model_name,
        help="HF embedding model name.",
    )
    args = parser.parse_args()

    results = search(args.query, Path(args.output), args.collection, args.top_k, args.model)
    for idx, item in enumerate(results, start=1):
        score = item.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"{idx}. {item['icon_path']} | {item['text']} | score={score_str}")


if __name__ == "__main__":
    main()
