from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("data")
    feedback_path: Path = Path("data/feedback/feedback.jsonl")
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    device: str = "cpu"
    weights_path: str | None = None
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-4
    margin: float = 0.2


def _load_metadata(data_dir: Path) -> list[dict]:
    metadata_path = data_dir / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_feedback(feedback_path: Path) -> list[dict]:
    records: list[dict] = []
    if not feedback_path.exists():
        return records
    with feedback_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_sign_image(sign_id: str, metadata: list[dict]) -> str | None:
    for entry in metadata:
        if entry.get("sign_id") == sign_id:
            return entry.get("image_path")
    return None


def _iter_triplets(metadata: list[dict], feedback: list[dict]) -> Iterable[tuple[str, str, str]]:
    if not feedback:
        return
    all_signs = [entry.get("sign_id") for entry in metadata if entry.get("sign_id")]
    for record in feedback:
        crop_path = record.get("crop_path")
        selected = record.get("selected", {})
        sign_id = selected.get("sign_id")
        pos_path = _resolve_sign_image(sign_id, metadata) if sign_id else None
        if not crop_path or not pos_path:
            continue

        candidates = [item.get("sign_id") for item in record.get("candidates", [])]
        negatives = [c for c in candidates if c and c != sign_id]
        if not negatives:
            negatives = [sid for sid in all_signs if sid != sign_id]
        neg_id = negatives[np.random.randint(0, len(negatives))]
        neg_path = _resolve_sign_image(neg_id, metadata)
        if not neg_path:
            continue
        yield crop_path, pos_path, neg_path


def _load_model(config: TrainConfig):
    try:
        import torch
        import timm

        model = timm.create_model(config.model_name, pretrained=True, num_classes=0)
        if config.weights_path:
            state = torch.load(config.weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        model.to(config.device)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return model, transform
    except Exception:
        import torch
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        if config.weights_path:
            state = torch.load(config.weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        model.to(config.device)
        transform = weights.transforms()
        return model, transform


def _extract_features(model, batch):
    if hasattr(model, "forward_features"):
        feats = model.forward_features(batch)
        if isinstance(feats, dict):
            feats = feats.get("x") or feats.get("x_norm_clstoken") or feats.get("x_norm_patchtokens")
        if feats is not None and feats.ndim > 2:
            feats = feats.mean(dim=1)
    else:
        feats = model(batch)
    return feats


def train(config: TrainConfig) -> None:
    import torch

    metadata = _load_metadata(config.data_dir)
    feedback = _load_feedback(config.feedback_path)
    triplets = list(_iter_triplets(metadata, feedback))
    if not triplets:
        raise RuntimeError("No feedback triplets found. Collect feedback first.")

    model, transform = _load_model(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = torch.nn.TripletMarginLoss(margin=config.margin)

    def _load_image(path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return transform(image)

    for epoch in range(config.epochs):
        np.random.shuffle(triplets)
        total_loss = 0.0
        for i in range(0, len(triplets), config.batch_size):
            batch = triplets[i : i + config.batch_size]
            anchors = torch.stack([_load_image(a) for a, _, _ in batch]).to(config.device)
            positives = torch.stack([_load_image(p) for _, p, _ in batch]).to(config.device)
            negatives = torch.stack([_load_image(n) for _, _, n in batch]).to(config.device)

            optimizer.zero_grad(set_to_none=True)
            anchor_feat = _extract_features(model, anchors)
            pos_feat = _extract_features(model, positives)
            neg_feat = _extract_features(model, negatives)

            anchor_feat = torch.nn.functional.normalize(anchor_feat, dim=1)
            pos_feat = torch.nn.functional.normalize(pos_feat, dim=1)
            neg_feat = torch.nn.functional.normalize(neg_feat, dim=1)

            loss = criterion(anchor_feat, pos_feat, neg_feat)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(triplets) // config.batch_size)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f}")

    output = config.data_dir / "finetuned.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved fine-tuned weights to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune embeddings from feedback.")
    parser.add_argument("--data", default="data", help="Data directory with metadata/index.")
    parser.add_argument(
        "--feedback",
        default="data/feedback/feedback.jsonl",
        help="Feedback jsonl path.",
    )
    parser.add_argument("--model", default=TrainConfig().model_name, help="Model name.")
    parser.add_argument("--device", default="cpu", help="Device, e.g. cpu or cuda.")
    parser.add_argument("--weights", default=None, help="Optional local weights path.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin.")
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=Path(args.data),
        feedback_path=Path(args.feedback),
        model_name=args.model,
        device=args.device,
        weights_path=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
    )
    train(config)


if __name__ == "__main__":
    main()
