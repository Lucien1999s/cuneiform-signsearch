from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: int = 224
    padding_ratio: float = 0.1
    background: int = 255


def _pad_to_square(image: Image.Image, padding_ratio: float, background: int) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    pad = int(max_dim * padding_ratio)
    canvas_size = max_dim + pad * 2
    canvas = Image.new("L", (canvas_size, canvas_size), color=background)
    offset = ((canvas_size - width) // 2, (canvas_size - height) // 2)
    canvas.paste(image, offset)
    return canvas


def normalize_image(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    padded = _pad_to_square(grayscale, config.padding_ratio, config.background)
    resized = padded.resize((config.target_size, config.target_size), Image.BICUBIC)
    return resized.convert("RGB")


def make_views(image: Image.Image) -> list[Image.Image]:
    views = [image]
    views.append(ImageOps.autocontrast(image))
    views.append(image.filter(ImageFilter.SHARPEN))
    views.append(image.filter(ImageFilter.GaussianBlur(radius=0.6)))
    views.append(ImageEnhance.Contrast(image).enhance(1.2))
    return views


def iter_views(image: Image.Image, config: PreprocessConfig) -> Iterable[Image.Image]:
    normalized = normalize_image(image, config)
    for view in make_views(normalized):
        yield view
