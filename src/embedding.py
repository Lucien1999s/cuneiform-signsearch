from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    device: str = "cpu"
    weights_path: str | None = None


class Embedder:
    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self.model, self.transform = self._load_model()

    def _load_model(self) -> tuple[object, Callable[[Image.Image], object]]:
        try:
            import torch
            import timm

            if self.config.weights_path:
                os.environ["TORCH_HOME"] = os.path.dirname(self.config.weights_path)

            model = timm.create_model(self.config.model_name, pretrained=True)
            if self.config.weights_path:
                state = torch.load(self.config.weights_path, map_location="cpu")
                model.load_state_dict(state, strict=False)
            model.eval()
            model.to(self.config.device)

            data_config = timm.data.resolve_model_data_config(model)
            transform = timm.data.create_transform(**data_config, is_training=False)
            return model, transform
        except Exception:
            try:
                import torch
                from torchvision.models import vit_b_16, ViT_B_16_Weights

                weights = ViT_B_16_Weights.DEFAULT
                model = vit_b_16(weights=weights)
                if self.config.weights_path:
                    state = torch.load(self.config.weights_path, map_location="cpu")
                    model.load_state_dict(state, strict=False)
                model.eval()
                model.to(self.config.device)
                transform = weights.transforms()
                return model, transform
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load an embedding model. Install timm/torch/torchvision "
                    "or provide local weights."
                ) from exc

    def _extract_features(self, batch: "torch.Tensor") -> "torch.Tensor":
        import torch

        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(batch)
            if isinstance(feats, dict):
                feats = (
                    feats.get("x_norm_clstoken")
                    or feats.get("x_norm_patchtokens")
                    or feats.get("x")
                )
            if feats.ndim > 2:
                feats = feats.mean(dim=1)
        else:
            feats = self.model(batch)
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        return feats

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        tensors = [self.transform(img).unsqueeze(0) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.config.device)
        with torch.no_grad():
            feats = self._extract_features(batch)
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def embed_batched(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            tensors = [self.transform(img).unsqueeze(0) for img in batch_images]
            batch = torch.cat(tensors, dim=0).to(self.config.device)
            with torch.no_grad():
                feats = self._extract_features(batch)
                feats = torch.nn.functional.normalize(feats, dim=1)
            outputs.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(outputs)
