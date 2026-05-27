from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_service.image_ops import auto_invert_dark_background, clip_bbox, mask_bboxes, pil_to_numpy, polygon_to_bbox, rotate_by_label
from ocr_service.pipeline import OCRPipeline

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}

LAYOUT_COLOR = (0, 102, 255)  # Blue
REGION_COLOR = (255, 0, 0)    # Red
TEXT_COLOR = (255, 165, 0)    # Orange


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR pipeline and save overlays for layout/region/text detections."
    )
    parser.add_argument(
        "--input",
        default="test_set",
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--output",
        default="data/ocr_box_overlays",
        help="Output directory for annotated images and metadata.",
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "full"),
        default=None,
        help="OCR runtime profile passed to OCRPipeline.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="Optional OCR engine override.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional OCR device override.",
    )
    return parser.parse_args()


def collect_inputs(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported input type: {input_path}")
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def preprocess_page_for_detection(pipeline: OCRPipeline, page_image: np.ndarray) -> Image.Image:
    image = Image.fromarray(page_image).convert("RGB")
    image, _ = auto_invert_dark_background(image)

    if pipeline.config.use_doc_orientation_classify:
        orientation = pipeline.runtime.predict_doc_orientation(pil_to_numpy(image))
        label = pipeline._extract_first_label(orientation)
        image = rotate_by_label(image, label)

    if pipeline.config.use_doc_unwarping:
        unwarp = pipeline.runtime.predict_doc_unwarp(pil_to_numpy(image))
        unwarp_img = unwarp.get("doctr_img")
        if isinstance(unwarp_img, np.ndarray) and unwarp_img.size:
            image = Image.fromarray(unwarp_img.astype(np.uint8)).convert("RGB")

    return image


def extract_layout_boxes(raw: dict) -> list[dict[str, object]]:
    boxes = OCRPipeline._extract_layout_boxes(raw)
    normalized: list[dict[str, object]] = []
    for box in boxes:
        normalized.append(
            {
                "bbox": box["bbox"],
                "label": str(box.get("label") or "layout"),
                "score": float(box.get("score") or 0.0),
            }
        )
    return normalized


def extract_text_boxes(raw: dict, width: int, height: int) -> list[list[int]]:
    polys = raw.get("dt_polys")
    if polys is None:
        return []
    if isinstance(polys, np.ndarray):
        polys = polys.tolist()

    boxes: list[list[int]] = []
    for poly in polys:
        bbox = clip_bbox(polygon_to_bbox(poly), width, height)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            boxes.append(bbox)
    return boxes


def draw_labeled_box(draw: ImageDraw.ImageDraw, bbox: list[int], label: str, color: tuple[int, int, int], *, width: int) -> None:
    draw.rectangle(tuple(bbox), outline=color, width=width)

    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    x1, y1, x2, _ = bbox
    label_x = max(0, min(x1, x2 - text_w - 4))
    label_y = max(0, y1 - text_h - 6)
    bg_box = (label_x, label_y, label_x + text_w + 4, label_y + text_h + 4)

    draw.rectangle(bg_box, fill=(255, 255, 255))
    draw.text((label_x + 2, label_y + 2), label, fill=color, font=font)


def draw_boxes(
    image: Image.Image,
    layout_boxes: list[dict[str, object]],
    region_boxes: list[dict[str, object]],
    text_boxes: list[list[int]],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for layout_item in layout_boxes:
        bbox = list(layout_item["bbox"])
        label = str(layout_item.get("label") or "layout")
        draw_labeled_box(draw, bbox, label, LAYOUT_COLOR, width=3)

    for region_item in region_boxes:
        bbox = list(region_item["bbox"])
        label = str(region_item.get("label") or "region")
        draw_labeled_box(draw, bbox, label, REGION_COLOR, width=2)

    for bbox in text_boxes:
        draw_labeled_box(draw, bbox, "text", TEXT_COLOR, width=2)

    return annotated


def build_output_paths(input_path: Path, input_root: Path, output_root: Path, page_index: int | None) -> tuple[Path, Path]:
    relative = input_path.relative_to(input_root) if input_root.is_dir() else Path(input_path.name)
    base = relative.with_suffix("")
    if page_index is not None:
        base = base.parent / f"{base.name}_page_{page_index + 1}"

    image_out = output_root / "images" / base.with_suffix(".png")
    metadata_out = output_root / "metadata" / base.with_suffix(".json")
    return image_out, metadata_out


def run(input_path: Path, output_root: Path, profile: str | None, engine: str | None, device: str | None) -> None:
    inputs = collect_inputs(input_path)
    if not inputs:
        raise FileNotFoundError(f"No supported inputs found in: {input_path}")

    input_root = input_path if input_path.is_dir() else input_path.parent

    pipeline = OCRPipeline(profile=profile, engine=engine, device=device)

    for file_path in inputs:
        # Run the full pipeline as requested.
        pipeline.predict_document(file_path)

        page_images = pipeline._load_page_images(file_path)
        for page_index, page_image in enumerate(page_images):
            working = preprocess_page_for_detection(pipeline, page_image)
            working_np = pil_to_numpy(working)
            width, height = working.size

            layout_raw = pipeline.runtime.predict_layout(working_np) if pipeline.config.use_layout_detection else {}
            region_raw: dict[str, object] = {}
            layout_boxes = extract_layout_boxes(layout_raw)
            region_boxes = extract_layout_boxes(region_raw)
            formula_items: list[dict[str, object]] = []
            masked_for_text = working
            if pipeline.config.use_formula_recognition:
                formula_boxes = [
                    list(item["bbox"])
                    for item in layout_boxes
                    if "formula" in str(item.get("label", "")).lower()
                ]
                if formula_boxes:
                    formula_items = pipeline._run_formula(working, formula_boxes)
                    if formula_items:
                        masked_for_text = mask_bboxes(
                            working,
                            [list(item["bbox"]) for item in formula_items if "bbox" in item],
                        )

            text_raw = pipeline.runtime.predict_text_detection(pil_to_numpy(masked_for_text))
            text_boxes = extract_text_boxes(text_raw, width, height)

            annotated = draw_boxes(working, layout_boxes, region_boxes, text_boxes)

            out_image_path, out_metadata_path = build_output_paths(
                file_path,
                input_root,
                output_root,
                page_index if len(page_images) > 1 else None,
            )
            out_image_path.parent.mkdir(parents=True, exist_ok=True)
            out_metadata_path.parent.mkdir(parents=True, exist_ok=True)

            annotated.save(out_image_path)

            metadata = {
                "source": str(file_path),
                "page_index": page_index,
                "image_size": [width, height],
                "counts": {
                    "layout": len(layout_boxes),
                    "region": len(region_boxes),
                    "formula": len(formula_items),
                    "text": len(text_boxes),
                },
                "boxes": {
                    "layout": [list(item["bbox"]) for item in layout_boxes],
                    "region": [list(item["bbox"]) for item in region_boxes],
                    "formula": [list(item["bbox"]) for item in formula_items if "bbox" in item],
                    "text": text_boxes,
                },
                "layout_detections": layout_boxes,
                "region_detections": region_boxes,
                "formula_detections": formula_items,
            }
            out_metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

            print(
                f"processed {file_path.name}"
                f" page={page_index + 1} layout={len(layout_boxes)} region={len(region_boxes)} text={len(text_boxes)}"
            )


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=Path(args.input).expanduser().resolve(),
        output_root=Path(args.output).expanduser().resolve(),
        profile=args.profile,
        engine=args.engine,
        device=args.device,
    )
