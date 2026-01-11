from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime, timezone
from typing import Any

import gradio as gr
from PIL import Image

from src.embedding import Embedder, EmbedderConfig
from src.preprocess import PreprocessConfig
from src.search import SearchConfig, default_embedder, search_signs


DATA_DIR = Path("data")
FEEDBACK_DIR = DATA_DIR / "feedback"


def _format_results(results: list[dict]) -> tuple[list[tuple[Image.Image, str]], list[list[Any]]]:
    gallery_items: list[tuple[Image.Image, str]] = []
    table_rows: list[list[Any]] = []

    for rank, result in enumerate(results, start=1):
        path = result.get("image_path")
        image = Image.open(path) if path else None
        caption = f"{result.get('sign_id')} | {result.get('english')}"
        if image is not None:
            gallery_items.append((image, caption))
        table_rows.append(
            [
                rank,
                result.get("sign_id"),
                result.get("english"),
                f"{result.get('score'):.4f}",
                path,
            ]
        )

    return gallery_items, table_rows


def _run_search(
    image: Image.Image | None,
    top_k: int,
    model_name: str,
    device: str,
    weights_path: str,
) -> tuple[list, list, list[dict]]:
    if image is None:
        return [], [], []

    weights = weights_path.strip() or None
    embedder = Embedder(
        EmbedderConfig(model_name=model_name, device=device, weights_path=weights)
    )
    results = search_signs(
        image,
        embedder,
        PreprocessConfig(),
        SearchConfig(top_k=top_k, data_dir=DATA_DIR),
    )
    gallery_items, table_rows = _format_results(results)
    return gallery_items, table_rows, results


def _select_gallery(evt: gr.SelectData, results: list[dict]) -> int:
    if results and evt.index is not None:
        return int(evt.index)
    return -1


def _save_feedback(
    image: Image.Image | None,
    results: list[dict],
    selected_index: int,
) -> str:
    if image is None or not results or selected_index < 0:
        return "No selection saved."

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    crop_path = FEEDBACK_DIR / f"crop_{timestamp}.jpg"
    image.save(crop_path)

    record = {
        "timestamp": timestamp,
        "crop_path": str(crop_path),
        "selected": results[selected_index],
        "candidates": results,
    }
    with (FEEDBACK_DIR / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return f"Saved selection: {results[selected_index].get('sign_id')}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Cuneiform Sign Visual Lookup") as demo:
        gr.Markdown("# Cuneiform Sign Visual Lookup (MVP)")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload and crop a sign",
                    tool="crop",
                    type="pil",
                )
                top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                model_name = gr.Textbox(
                    label="Model name",
                    value=EmbedderConfig().model_name,
                )
                device = gr.Dropdown(
                    ["cpu", "cuda"],
                    value="cpu",
                    label="Device",
                )
                weights_path = gr.Textbox(
                    label="Local weights path (optional)",
                    value="",
                )
                search_btn = gr.Button("Search")
                save_btn = gr.Button("Save Selection")
                status = gr.Markdown("")

            with gr.Column():
                gallery = gr.Gallery(label="Candidates", columns=3, height=360)
                table = gr.Dataframe(
                    headers=["rank", "sign_id", "english", "score", "image_path"],
                    interactive=False,
                )

        results_state = gr.State([])
        selected_state = gr.State(-1)

        search_btn.click(
            _run_search,
            inputs=[image_input, top_k, model_name, device, weights_path],
            outputs=[gallery, table, results_state],
        )
        gallery.select(
            _select_gallery,
            inputs=[results_state],
            outputs=[selected_state],
        )
        save_btn.click(
            _save_feedback,
            inputs=[image_input, results_state, selected_state],
            outputs=[status],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
