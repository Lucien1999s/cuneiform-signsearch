from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import gradio as gr

from src.text_search import search


ASSETS_ROOT = Path("datasets")


def _image_to_data_uri(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    data = file_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _render_cards(results: list[dict[str, Any]]) -> str:
    if not results:
        return "<div class='empty'>No results yet.</div>"

    cards = []
    for item in results:
        score = item.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        icon_path = item.get("icon_path") or ""
        img_src = _image_to_data_uri(icon_path)
        text = item.get("text") or ""
        cards.append(
            f"""
            <div class="card">
              <div class="thumb">
                <img src="{img_src}" alt="cuneiform tablet"/>
              </div>
              <div class="meta">
                <div class="text">{text}</div>
                <div class="score">Score: {score_str}</div>
              </div>
            </div>
            """
        )
    return "<div class='grid'>" + "".join(cards) + "</div>"


def _run_search(query: str, top_k: int) -> str:
    query = (query or "").strip()
    if not query:
        return "<div class='empty'>Enter a query to search.</div>"

    results = search(
        query=query,
        output_dir=Path("data"),
        collection_name="tablets",
        top_k=top_k,
        model_name="BAAI/bge-large-en-v1.5",
    )
    return _render_cards(results)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Cuneiform Tablet Semantic Search") as demo:
        gr.HTML(
            """
            <div class="hero">
              <div class="title">Cuneiform Tablet Semantic Search</div>
              <div class="subtitle">
                Enter English words to retrieve semantically related cuneiform tablets.
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                query = gr.Textbox(
                    label="English Query",
                    placeholder="Try: barley trade, temple offerings, land purchase, messenger, tax record",
                )
            with gr.Column(scale=1, min_width=160):
                top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                run = gr.Button("Search", variant="primary")

        results = gr.HTML()

        run.click(_run_search, inputs=[query, top_k], outputs=[results])

        gr.HTML(
            """
            <div class="footer">
              Index source: translated cuneiform sign glossary. Model: BAAI/bge-large-en-v1.5.
            </div>
            """
        )

    demo.css = """
    :root {
      --paper: #f7f3ec;
      --ink: #1f1f1a;
      --accent: #c9772b;
      --accent-dark: #8f541f;
      --card: #ffffff;
      --muted: #5f5a4f;
    }
    body, .gradio-container {
      background: radial-gradient(1200px 600px at 5% 0%, #fff8ec 0%, #f2eadf 40%, #efe5d7 100%);
      color: var(--ink);
      font-family: "Baskerville", "Georgia", "Times New Roman", serif;
    }
    .hero {
      padding: 22px 0 12px 0;
      border-bottom: 1px solid rgba(0,0,0,0.06);
      margin-bottom: 10px;
      animation: fadeUp 420ms ease-out;
    }
    .title {
      font-size: 44px;
      font-weight: 800;
      letter-spacing: 0.6px;
      text-transform: uppercase;
    }
    .subtitle {
      color: var(--muted);
      margin-top: 8px;
      font-size: 16px;
      max-width: 760px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 18px;
    }
    .card {
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 16px;
      overflow: hidden;
      background: var(--card);
      box-shadow: 0 10px 24px rgba(0,0,0,0.08);
      transform: translateY(0);
      transition: transform 200ms ease, box-shadow 200ms ease;
      animation: fadeUp 500ms ease-out;
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 16px 28px rgba(0,0,0,0.12);
    }
    .thumb {
      height: 190px;
      background: repeating-linear-gradient(
        135deg,
        #f7efe4 0px,
        #f7efe4 10px,
        #f3e7d8 10px,
        #f3e7d8 20px
      );
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .thumb img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));
    }
    .meta {
      padding: 14px 16px 16px 16px;
    }
    .text {
      font-size: 15px;
      line-height: 1.45;
      min-height: 46px;
      font-weight: 700;
    }
    .score {
      margin-top: 10px;
      font-size: 12px;
      color: var(--accent-dark);
      font-weight: 800;
      letter-spacing: 0.4px;
    }
    .footer {
      margin-top: 20px;
      color: var(--muted);
      font-size: 12px;
    }
    .empty {
      color: var(--muted);
      font-size: 14px;
      padding: 6px 0;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    """
    return demo


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
