from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab helper for training.")
    parser.add_argument("--install", action="store_true", help="Install requirements.")
    parser.add_argument("--data", default="data", help="Data directory.")
    parser.add_argument(
        "--feedback",
        default="data/feedback/feedback.jsonl",
        help="Feedback jsonl path.",
    )
    parser.add_argument("--model", default=None, help="Model name override.")
    parser.add_argument("--device", default=None, help="Device override.")
    parser.add_argument("--weights", default=None, help="Optional local weights path.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    if args.install:
        _run([sys.executable, "-m", "pip", "install", "-r", str(repo_root / "requirements.txt")])

    device = args.device or _detect_device()
    cmd = [
        sys.executable,
        str(repo_root / "src" / "train.py"),
        "--data",
        args.data,
        "--feedback",
        args.feedback,
        "--device",
        device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--margin",
        str(args.margin),
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.weights:
        cmd += ["--weights", args.weights]
    _run(cmd)


if __name__ == "__main__":
    main()
