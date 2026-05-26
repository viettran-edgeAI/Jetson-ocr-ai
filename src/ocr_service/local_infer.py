from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import default_model_dir as _default_model_dir
from .models import OCRResult
from .pipeline import OCRPipeline


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


def _collect_inputs(input_path: Path) -> tuple[list[Path], Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported input type: {input_path}")
        return [input_path], input_path.parent

    inputs = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return inputs, input_path


def _clear_tensorrt_caches(model_dirs: Iterable[Path]) -> None:
    for model_dir in model_dirs:
        cache_dir = model_dir / ".cache" / "paddle"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def _annotate_image(image_path: Path, result: OCRResult, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for line in result.lines:
        if not line.polygon:
            continue
        points = [tuple(point) for point in line.polygon]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=(0, 255, 0), width=2)
        if line.text:
            x, y = points[0]
            draw.text((x, max(0, y - 12)), line.text, fill=(255, 32, 32), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_markdown(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def _print_result(image_path: Path, result: OCRResult) -> None:
    print(f"<!-- source: {image_path} -->")
    print(result.markdown_text)


def _resolve_output_paths(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    page_index: int | None = None,
) -> tuple[Path, Path]:
    if input_root.is_dir():
        relative = image_path.relative_to(input_root)
    else:
        relative = image_path.name

    base = Path(relative).with_suffix("")
    if page_index is not None:
        base = base.parent / f"{base.name}_page_{page_index + 1}"
    markdown_path = output_root / "markdown" / base.with_suffix(".md")
    image_out_path = output_root / "images" / base.with_suffix(".png")
    return markdown_path, image_out_path


def _resolve_document_output_path(image_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_dir():
        relative = image_path.relative_to(input_root)
    else:
        relative = image_path.name
    base = Path(relative).with_suffix(".md")
    return output_root / "documents" / base


def run_local_infer(
    input_path: Path,
    output_root: Path,
    pipeline: OCRPipeline,
    write_visuals: bool,
) -> list[OCRResult]:
    inputs, input_root = _collect_inputs(input_path)
    if not inputs:
        raise FileNotFoundError(f"No supported inputs found in: {input_path}")

    results: list[OCRResult] = []
    for image_path in inputs:
        document_results = pipeline.predict_document(image_path)
        document_markdown = pipeline.build_document_markdown(
            document_results,
            original_filename=image_path.name,
            content_type="application/pdf" if image_path.suffix.lower() == ".pdf" else None,
        )
        _write_markdown(
            document_markdown,
            _resolve_document_output_path(image_path, input_root, output_root),
        )
        for result in document_results:
            _print_result(image_path, result)
            page_index = result.meta.get("page_index")
            markdown_path, image_out_path = _resolve_output_paths(
                image_path,
                input_root,
                output_root,
                page_index=page_index,
            )
            _write_markdown(result.markdown_text + "\n", markdown_path)
            if write_visuals and image_path.suffix.lower() != ".pdf":
                _annotate_image(image_path, result, image_out_path)
            results.append(result)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Jetson OCR pipeline on images.")
    parser.add_argument(
        "input",
        help="Image file or directory to process.",
    )
    parser.add_argument(
        "--output",
        default="data/ocr_local",
        help="Output folder for Markdown and annotated images.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Disable annotated image output.",
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "full"),
        default=None,
        help="OCR runtime profile. 'fast' disables the three optional preprocessing stages; 'full' enables them.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="Override OCR_ENGINE for this run (default runtime recommendation: paddle_static).",
    )
    parser.add_argument(
        "--text-recognition-batch-size",
        type=int,
        default=None,
        help="Override OCR_TEXT_RECOGNITION_BATCH_SIZE for this run.",
    )
    parser.add_argument(
        "--textline-orientation-batch-size",
        type=int,
        default=None,
        help="Override OCR_TEXTLINE_ORI_BATCH_SIZE for this run.",
    )
    parser.add_argument(
        "--formula-recognition-batch-size",
        type=int,
        default=None,
        help="Override OCR_FORMULA_RECOGNITION_BATCH_SIZE for this run.",
    )
    parser.add_argument("--device", default=None, help="Override OCR_DEVICE for this run.")
    parser.add_argument("--det-model-dir", default=None, help="Override OCR_DET_MODEL_DIR for this run.")
    parser.add_argument("--rec-model-dir", default=None, help="Override OCR_REC_MODEL_DIR for this run.")
    parser.add_argument("--layout-model-dir", default=None, help="Override OCR_LAYOUT_MODEL_DIR for this run.")
    parser.add_argument("--region-model-dir", default=None, help="Override OCR_REGION_MODEL_DIR for this run.")
    parser.add_argument("--formula-model-dir", default=None, help="Override OCR_FORMULA_MODEL_DIR for this run.")
    parser.add_argument(
        "--use-document-structure",
        action="store_true",
        help="Enable the PP-StructureV3 layout, optional region, formula, masking, and merge path.",
    )
    parser.add_argument(
        "--use-tensorrt",
        action="store_true",
        help="Enable the Jetson TensorRT profile for selected OCR modules.",
    )
    parser.add_argument(
        "--trt-profile",
        default=None,
        help="TensorRT profile name to record in metadata (default: jetson).",
    )
    parser.add_argument(
        "--trt-modules",
        default=None,
        help="Comma-separated TensorRT module list: det,rec,textline,doc_ori,doc_unwarp,all (default: det,rec).",
    )
    parser.add_argument(
        "--clear-trt-cache",
        action="store_true",
        help="Clear Paddle-TRT cache directories for local OCR models before running.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run this many unrecorded predictions on the first input before writing measured outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    trt_modules = None
    if args.trt_modules:
        trt_modules = tuple(part.strip().lower() for part in args.trt_modules.split(",") if part.strip())

    if args.clear_trt_cache:
        model_dirs = [
            Path(args.det_model_dir).expanduser().resolve()
            if args.det_model_dir
            else _default_model_dir("PP-OCRv5_mobile_det_infer"),
            Path(args.rec_model_dir).expanduser().resolve()
            if args.rec_model_dir
            else _default_model_dir("PP-OCRv5_mobile_rec_infer"),
            _default_model_dir("PP-LCNet_x0_25_textline_ori_infer"),
            _default_model_dir("PP-LCNet_x1_0_doc_ori_infer"),
            _default_model_dir("UVDoc_infer"),
            _default_model_dir("PP-DocLayout_plus-L_infer"),
            _default_model_dir("PP-DocBlockLayout_infer"),
            _default_model_dir("PP-FormulaNet_plus-S_infer"),
        ]
        _clear_tensorrt_caches(model_dirs)

    if args.use_document_structure:
        os.environ["OCR_USE_DOCUMENT_STRUCTURE"] = "1"

    pipeline = OCRPipeline(
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        device=args.device,
        layout_detection_model_dir=args.layout_model_dir,
        region_detection_model_dir=args.region_model_dir,
        formula_recognition_model_dir=args.formula_model_dir,
        profile=args.profile,
        engine=args.engine,
        text_recognition_batch_size=args.text_recognition_batch_size,
        textline_orientation_batch_size=args.textline_orientation_batch_size,
        formula_recognition_batch_size=args.formula_recognition_batch_size,
        use_tensorrt=args.use_tensorrt or None,
        trt_profile=args.trt_profile,
        trt_modules=trt_modules,
    )

    if args.warmup_runs > 0:
        warmup_inputs, _ = _collect_inputs(input_path)
        if warmup_inputs:
            for _ in range(args.warmup_runs):
                pipeline.predict_document(warmup_inputs[0])

    run_local_infer(
        input_path=input_path,
        output_root=output_root,
        pipeline=pipeline,
        write_visuals=not args.no_visuals,
    )


if __name__ == "__main__":
    main()
