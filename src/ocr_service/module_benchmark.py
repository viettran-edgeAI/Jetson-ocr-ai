from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from paddleocr._models.doc_img_orientation_classification import (
    DocImgOrientationClassification,
)
from paddleocr._models.text_detection import TextDetection
from paddleocr._models.text_image_unwarping import TextImageUnwarping
from paddleocr._models.text_recognition import TextRecognition
from paddleocr._models.textline_orientation_classification import (
    TextLineOrientationClassification,
)
from paddlex.inference.pipelines.components import CropByPolys, SortQuadBoxes, rotate_image

from .config import default_model_dir as _default_model_dir
from .config import env_int as _env_int
from .config import env_str as _env_str


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class ModuleRun:
    doc_orientation_ms: float
    doc_unwarping_ms: float
    text_detection_ms: float
    textline_orientation_ms: float
    text_recognition_ms: float
    helper_ms: float
    total_ms: float
    doc_angle: int
    det_boxes: int
    line_orientation_boxes: int
    recognized_lines: int
    textline_orientation_batches: int
    text_recognition_batches: int
    text_recognition_ms_per_line: float
    textline_orientation_ms_per_box: float
    crop_ratio_min: float
    crop_ratio_p50: float
    crop_ratio_p95: float
    crop_ratio_max: float
    rec_input_width_min: int
    rec_input_width_p50: int
    rec_input_width_p95: int
    rec_input_width_max: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_orientation_ms": round(self.doc_orientation_ms, 2),
            "doc_unwarping_ms": round(self.doc_unwarping_ms, 2),
            "text_detection_ms": round(self.text_detection_ms, 2),
            "textline_orientation_ms": round(self.textline_orientation_ms, 2),
            "text_recognition_ms": round(self.text_recognition_ms, 2),
            "helper_ms": round(self.helper_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "doc_angle": self.doc_angle,
            "det_boxes": self.det_boxes,
            "line_orientation_boxes": self.line_orientation_boxes,
            "recognized_lines": self.recognized_lines,
            "textline_orientation_batches": self.textline_orientation_batches,
            "text_recognition_batches": self.text_recognition_batches,
            "text_recognition_ms_per_line": round(self.text_recognition_ms_per_line, 2),
            "textline_orientation_ms_per_box": round(self.textline_orientation_ms_per_box, 2),
            "crop_ratio_min": round(self.crop_ratio_min, 2),
            "crop_ratio_p50": round(self.crop_ratio_p50, 2),
            "crop_ratio_p95": round(self.crop_ratio_p95, 2),
            "crop_ratio_max": round(self.crop_ratio_max, 2),
            "rec_input_width_min": self.rec_input_width_min,
            "rec_input_width_p50": self.rec_input_width_p50,
            "rec_input_width_p95": self.rec_input_width_p95,
            "rec_input_width_max": self.rec_input_width_max,
        }


def _collect_images(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {input_path}")
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_image_bgr(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _maybe_float(value: Any) -> float:
    return round(float(value), 2)


def _batch_count(item_count: int, batch_size: int) -> int:
    if item_count <= 0:
        return 0
    return int(ceil(item_count / float(batch_size)))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _recognition_input_width(crop: np.ndarray, image_height: int = 48, default_width: int = 320) -> int:
    if crop.size <= 0 or crop.shape[0] <= 0:
        return default_width

    ratio = crop.shape[1] / float(crop.shape[0])
    return min(3200, max(default_width, int(ceil(image_height * ratio))))


class OCRModuleBenchmark:
    def __init__(
        self,
        *,
        device: str | None = None,
        engine: str | None = None,
        text_recognition_batch_size: int | None = None,
        textline_orientation_batch_size: int | None = None,
    ) -> None:
        self.device = device or _env_str(("OCR_DEVICE",), "gpu:0")
        self.engine = engine or _env_str(("OCR_ENGINE",), "paddle_static")
        self.text_det_limit_side_len = _env_int(
            ("OCR_TEXT_DET_LIMIT_SIDE_LEN", "OCR_DET_LIMIT_SIDE_LEN"),
            64,
        )
        self.text_det_limit_type = _env_str(
            ("OCR_TEXT_DET_LIMIT_TYPE", "OCR_DET_LIMIT_TYPE"),
            "min",
        )
        self.text_recognition_batch_size = text_recognition_batch_size or _env_int(
            ("OCR_TEXT_RECOGNITION_BATCH_SIZE", "OCR_REC_BATCH_SIZE"),
            16,
        )
        self.textline_orientation_batch_size = textline_orientation_batch_size or _env_int(
            ("OCR_TEXTLINE_ORI_BATCH_SIZE",),
            6,
        )

        self.doc_orientation_model = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori",
            model_dir=str(_default_model_dir("PP-LCNet_x1_0_doc_ori_infer")),
            device=self.device,
            engine=self.engine,
        )
        self.doc_unwarping_model = TextImageUnwarping(
            model_name="UVDoc",
            model_dir=str(_default_model_dir("UVDoc_infer")),
            device=self.device,
            engine=self.engine,
        )
        self.text_detection_model = TextDetection(
            model_name="PP-OCRv5_mobile_det",
            model_dir=str(_default_model_dir("PP-OCRv5_mobile_det")),
            device=self.device,
            engine=self.engine,
            limit_side_len=self.text_det_limit_side_len,
            limit_type=self.text_det_limit_type,
        )
        self.textline_orientation_model = TextLineOrientationClassification(
            model_name="PP-LCNet_x0_25_textline_ori",
            model_dir=str(_default_model_dir("PP-LCNet_x0_25_textline_ori_infer")),
            device=self.device,
            engine=self.engine,
        )
        self.text_recognition_model = TextRecognition(
            model_name="PP-OCRv5_mobile_rec",
            model_dir=str(_default_model_dir("PP-OCRv5_mobile_rec")),
            device=self.device,
            engine=self.engine,
        )
        self._sort_boxes = SortQuadBoxes()
        self._crop_by_polys = CropByPolys(det_box_type="quad")

    def run_once(self, image_path: Path) -> ModuleRun:
        image_bgr = _load_image_bgr(image_path)
        start_total = perf_counter()

        start_doc_orientation = perf_counter()
        doc_orientation_result = self.doc_orientation_model.predict([image_bgr])[0]
        doc_orientation_ms = (perf_counter() - start_doc_orientation) * 1000.0

        doc_angle = int(doc_orientation_result["label_names"][0])
        rotated_doc = rotate_image(image_bgr, doc_angle)

        start_doc_unwarping = perf_counter()
        doc_unwarping_result = self.doc_unwarping_model.predict([rotated_doc])[0]
        doc_unwarping_ms = (perf_counter() - start_doc_unwarping) * 1000.0

        # PaddleX returns RGB here; OCR pipeline flips it back to BGR before detection.
        unwarped_bgr = np.asarray(doc_unwarping_result["doctr_img"])[:, :, ::-1].copy()

        start_text_detection = perf_counter()
        text_detection_result = self.text_detection_model.predict(
            [unwarped_bgr],
            limit_side_len=self.text_det_limit_side_len,
            limit_type=self.text_det_limit_type,
            batch_size=1,
        )[0]
        text_detection_ms = (perf_counter() - start_text_detection) * 1000.0

        dt_polys = self._sort_boxes(text_detection_result["dt_polys"])
        cropped_lines = list(self._crop_by_polys(unwarped_bgr, dt_polys))

        filtered_crops: list[np.ndarray] = []
        filtered_polys: list[Any] = []
        for crop, poly in zip(cropped_lines, dt_polys):
            if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                filtered_crops.append(crop)
                filtered_polys.append(poly)

        crop_ratios = [crop.shape[1] / float(crop.shape[0]) for crop in filtered_crops]
        rec_input_widths = [_recognition_input_width(crop) for crop in filtered_crops]

        start_textline_orientation = perf_counter()
        if filtered_crops:
            textline_orientation_results = self.textline_orientation_model.predict(
                filtered_crops,
                batch_size=self.textline_orientation_batch_size,
            )
            line_orientation_angles = [
                int(np.asarray(item["class_ids"], dtype=np.int64).ravel()[0])
                for item in textline_orientation_results
            ]
            rotated_line_crops = [
                rotate_image(crop, angle * 180)
                for crop, angle in zip(filtered_crops, line_orientation_angles)
            ]
        else:
            line_orientation_angles = []
            rotated_line_crops = []
        textline_orientation_ms = (perf_counter() - start_textline_orientation) * 1000.0

        start_text_recognition = perf_counter()
        recognized_lines = 0
        if rotated_line_crops:
            indexed_crops = [
                {"crop_id": idx, "ratio": crop.shape[1] / float(crop.shape[0]), "crop": crop}
                for idx, crop in enumerate(rotated_line_crops)
            ]
            sorted_crops = [item["crop"] for item in sorted(indexed_crops, key=lambda item: item["ratio"])]
            text_recognition_results = self.text_recognition_model.predict(
                sorted_crops,
                batch_size=self.text_recognition_batch_size,
                return_word_box=False,
            )
            recognized_lines = len(text_recognition_results)
        text_recognition_ms = (perf_counter() - start_text_recognition) * 1000.0

        total_ms = (perf_counter() - start_total) * 1000.0
        helper_ms = total_ms - (
            doc_orientation_ms
            + doc_unwarping_ms
            + text_detection_ms
            + textline_orientation_ms
            + text_recognition_ms
        )

        return ModuleRun(
            doc_orientation_ms=doc_orientation_ms,
            doc_unwarping_ms=doc_unwarping_ms,
            text_detection_ms=text_detection_ms,
            textline_orientation_ms=textline_orientation_ms,
            text_recognition_ms=text_recognition_ms,
            helper_ms=helper_ms,
            total_ms=total_ms,
            doc_angle=doc_angle,
            det_boxes=len(filtered_polys),
            line_orientation_boxes=len(line_orientation_angles),
            recognized_lines=recognized_lines,
            textline_orientation_batches=_batch_count(
                len(line_orientation_angles),
                self.textline_orientation_batch_size,
            ),
            text_recognition_batches=_batch_count(recognized_lines, self.text_recognition_batch_size),
            text_recognition_ms_per_line=text_recognition_ms / recognized_lines if recognized_lines else 0.0,
            textline_orientation_ms_per_box=(
                textline_orientation_ms / len(line_orientation_angles) if line_orientation_angles else 0.0
            ),
            crop_ratio_min=min(crop_ratios) if crop_ratios else 0.0,
            crop_ratio_p50=_percentile(crop_ratios, 0.50),
            crop_ratio_p95=_percentile(crop_ratios, 0.95),
            crop_ratio_max=max(crop_ratios) if crop_ratios else 0.0,
            rec_input_width_min=min(rec_input_widths) if rec_input_widths else 0,
            rec_input_width_p50=int(round(_percentile([float(width) for width in rec_input_widths], 0.50))),
            rec_input_width_p95=int(round(_percentile([float(width) for width in rec_input_widths], 0.95))),
            rec_input_width_max=max(rec_input_widths) if rec_input_widths else 0,
        )


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _summarize_runs(runs: list[ModuleRun]) -> dict[str, Any]:
    total_recognized_lines = sum(run.recognized_lines for run in runs)
    total_line_orientation_boxes = sum(run.line_orientation_boxes for run in runs)
    return {
        "doc_orientation_ms": _maybe_float(_mean([run.doc_orientation_ms for run in runs])),
        "doc_unwarping_ms": _maybe_float(_mean([run.doc_unwarping_ms for run in runs])),
        "text_detection_ms": _maybe_float(_mean([run.text_detection_ms for run in runs])),
        "textline_orientation_ms": _maybe_float(_mean([run.textline_orientation_ms for run in runs])),
        "text_recognition_ms": _maybe_float(_mean([run.text_recognition_ms for run in runs])),
        "helper_ms": _maybe_float(_mean([run.helper_ms for run in runs])),
        "total_ms": _maybe_float(_mean([run.total_ms for run in runs])),
        "doc_angle": int(round(_mean([run.doc_angle for run in runs]))),
        "det_boxes": int(round(_mean([run.det_boxes for run in runs]))),
        "line_orientation_boxes": int(round(_mean([run.line_orientation_boxes for run in runs]))),
        "recognized_lines": int(round(_mean([run.recognized_lines for run in runs]))),
        "textline_orientation_batches": int(round(_mean([run.textline_orientation_batches for run in runs]))),
        "text_recognition_batches": int(round(_mean([run.text_recognition_batches for run in runs]))),
        "text_recognition_ms_per_line": _maybe_float(
            sum(run.text_recognition_ms for run in runs) / total_recognized_lines
            if total_recognized_lines
            else 0.0
        ),
        "textline_orientation_ms_per_box": _maybe_float(
            sum(run.textline_orientation_ms for run in runs) / total_line_orientation_boxes
            if total_line_orientation_boxes
            else 0.0
        ),
        "crop_ratio_min": _maybe_float(_mean([run.crop_ratio_min for run in runs])),
        "crop_ratio_p50": _maybe_float(_mean([run.crop_ratio_p50 for run in runs])),
        "crop_ratio_p95": _maybe_float(_mean([run.crop_ratio_p95 for run in runs])),
        "crop_ratio_max": _maybe_float(_mean([run.crop_ratio_max for run in runs])),
        "rec_input_width_min": int(round(_mean([run.rec_input_width_min for run in runs]))),
        "rec_input_width_p50": int(round(_mean([run.rec_input_width_p50 for run in runs]))),
        "rec_input_width_p95": int(round(_mean([run.rec_input_width_p95 for run in runs]))),
        "rec_input_width_max": int(round(_mean([run.rec_input_width_max for run in runs]))),
    }


def benchmark_images(
    input_path: Path,
    output_path: Path,
    *,
    repeat: int,
    warmup: int,
    device: str | None,
    engine: str | None,
    text_recognition_batch_size: int | None,
    textline_orientation_batch_size: int | None,
) -> dict[str, Any]:
    images = _collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    runner = OCRModuleBenchmark(
        device=device,
        engine=engine,
        text_recognition_batch_size=text_recognition_batch_size,
        textline_orientation_batch_size=textline_orientation_batch_size,
    )

    image_payloads: list[dict[str, Any]] = []
    all_runs: list[ModuleRun] = []
    for image_path in images:
        for _ in range(warmup):
            runner.run_once(image_path)

        runs = [runner.run_once(image_path) for _ in range(repeat)]
        all_runs.extend(runs)

        with Image.open(image_path) as image:
            width, height = image.size

        image_payloads.append(
            {
                "image_path": str(image_path),
                "image_size": [width, height],
                "runs": [run.to_dict() for run in runs],
                "average": _summarize_runs(runs),
            }
        )

    summary = _summarize_runs(all_runs)
    payload = {
        "config": {
            "device": runner.device,
            "engine": runner.engine,
            "repeat": repeat,
            "warmup": warmup,
            "text_det_limit_side_len": runner.text_det_limit_side_len,
            "text_det_limit_type": runner.text_det_limit_type,
            "text_recognition_batch_size": runner.text_recognition_batch_size,
            "textline_orientation_batch_size": runner.textline_orientation_batch_size,
        },
        "images": image_payloads,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the five PP-OCRv5 component models individually.",
    )
    parser.add_argument(
        "input",
        help="Image file or directory to benchmark.",
    )
    parser.add_argument(
        "--output",
        default="data/ocr_module_benchmark/module_timings.json",
        help="Output JSON file for per-module timing results.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of timed runs per image after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs per image before timing.",
    )
    parser.add_argument("--device", default=None, help="Override OCR_DEVICE for this run.")
    parser.add_argument("--engine", default=None, help="Override OCR_ENGINE for this run.")
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    benchmark_images(
        input_path=input_path,
        output_path=output_path,
        repeat=args.repeat,
        warmup=args.warmup,
        device=args.device,
        engine=args.engine,
        text_recognition_batch_size=args.text_recognition_batch_size,
        textline_orientation_batch_size=args.textline_orientation_batch_size,
    )


if __name__ == "__main__":
    main()
