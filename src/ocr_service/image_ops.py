from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(ensure_rgb(image))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def auto_invert_dark_background(image: Image.Image) -> tuple[Image.Image, dict[str, float | bool]]:
    rgb = ensure_rgb(image)
    gray = np.asarray(rgb.convert("L"))
    height, width = gray.shape
    border = max(2, min(height, width) // 20)
    border_pixels = np.concatenate(
        (
            gray[:border, :].reshape(-1),
            gray[-border:, :].reshape(-1),
            gray[:, :border].reshape(-1),
            gray[:, -border:].reshape(-1),
        )
    )

    border_median = float(np.median(border_pixels))
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))
    p99 = float(np.percentile(gray, 99))
    mean = float(np.mean(gray))
    should_invert = border_median < 96.0 and p99 > 150.0 and (p99 - p10) > 50.0
    metadata = {
        "applied": should_invert,
        "border_median": border_median,
        "mean": mean,
        "p10": p10,
        "p90": p90,
        "p99": p99,
    }
    if not should_invert:
        return rgb, metadata
    return ImageOps.invert(rgb), metadata


def rotate_by_label(image: Image.Image, label: str | None) -> Image.Image:
    if label is None:
        return image
    angle = str(label).replace("°", "").strip()
    if angle == "90":
        return image.rotate(-90, expand=True)
    if angle == "180":
        return image.rotate(180, expand=True)
    if angle == "270":
        return image.rotate(90, expand=True)
    return image


def polygon_to_bbox(poly: Iterable[Iterable[float]]) -> list[int]:
    xs: list[float] = []
    ys: list[float] = []
    for point in poly:
        if len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    if not xs or not ys:
        return [0, 0, 0, 0]
    return [
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys)),
    ]


def bbox_to_polygon(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def clip_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def crop_by_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    return image.crop(tuple(bbox))


def mask_bboxes(image: Image.Image, bboxes: Iterable[list[int]]) -> Image.Image:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for bbox in bboxes:
        draw.rectangle(tuple(bbox), fill=(255, 255, 255))
    return masked


def sort_reading_order(items: list[dict]) -> list[dict]:
    if not items:
        return []

    def _height(item: dict) -> int:
        x1, y1, x2, y2 = item.get("bbox") or [0, 0, 0, 0]
        return max(1, int(y2) - int(y1))

    heights = sorted(_height(item) for item in items)
    median_height = heights[len(heights) // 2]
    line_tolerance = max(8, int(median_height * 0.7))

    def _key(item: dict) -> tuple[int, int, int]:
        x1, y1, x2, y2 = item.get("bbox") or [0, 0, 0, 0]
        center_y = (int(y1) + int(y2)) // 2
        return center_y // line_tolerance, int(x1), int(y1)

    return sorted(items, key=_key)
