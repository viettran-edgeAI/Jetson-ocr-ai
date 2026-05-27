from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .config import OCRRuntimeConfig
from .image_ops import (
    auto_invert_dark_background,
    bbox_to_polygon,
    clip_bbox,
    crop_by_bbox,
    mask_bboxes,
    pil_to_numpy,
    polygon_to_bbox,
    rotate_by_label,
    sort_reading_order,
)
from .models import OCRBlock, OCRFormula, OCRLine, OCRRegion, OCRResult
from .paddle_adapter import PaddleRuntime

_SINGLE_CHAR_ALLOWED = {
    "\n",
    "a",
    "b",
    "c",
    "d",
    "e",
    "A",
    "B",
    "C",
    "D",
    "E",
    "$",
    "\\",
    "[",
    "]",
    "(",
    ")",
    "{",
    "}",
}


@dataclass(slots=True)
class PipelineOptions:
    use_doc_orientation: bool = True
    use_doc_unwarp: bool = True
    use_layout: bool = True
    use_region: bool = True
    use_formula: bool = True
    use_textline_orientation: bool = True
    include_debug: bool = False


class OCRPipeline:
    def __init__(
        self,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        device: str | None = None,
        doc_orientation_model_dir: str | None = None,
        doc_unwarping_model_dir: str | None = None,
        textline_orientation_model_dir: str | None = None,
        layout_detection_model_dir: str | None = None,
        region_detection_model_dir: str | None = None,
        formula_recognition_model_dir: str | None = None,
        profile: str | None = None,
        engine: str | None = None,
        text_recognition_batch_size: int | None = None,
        textline_orientation_batch_size: int | None = None,
        formula_recognition_batch_size: int | None = None,
        enable_hpi: bool | None = None,
        use_tensorrt: bool | None = None,
        trt_profile: str | None = None,
        trt_modules: tuple[str, ...] | None = None,
    ) -> None:
        self.config = OCRRuntimeConfig.from_env(
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            device=device,
            doc_orientation_model_dir=doc_orientation_model_dir,
            doc_unwarping_model_dir=doc_unwarping_model_dir,
            textline_orientation_model_dir=textline_orientation_model_dir,
            layout_detection_model_dir=layout_detection_model_dir,
            region_detection_model_dir=region_detection_model_dir,
            formula_recognition_model_dir=formula_recognition_model_dir,
            profile=profile,
            engine=engine,
            text_recognition_batch_size=text_recognition_batch_size,
            textline_orientation_batch_size=textline_orientation_batch_size,
            formula_recognition_batch_size=formula_recognition_batch_size,
            enable_hpi=enable_hpi,
            use_tensorrt=use_tensorrt,
            trt_profile=trt_profile,
            trt_modules=trt_modules,
        )
        self.runtime = PaddleRuntime(self.config)

    def warmup_formula_module(self) -> None:
        if not self.config.use_formula_recognition:
            return
        image = Image.new("RGB", (960, 320), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((48, 56), "FORMULA WARMUP", fill=(0, 0, 0))
        draw.text((48, 156), r"E = mc^2  int_0^1 x^2 dx = 1/3", fill=(0, 0, 0))
        self._run_formula(image, [[32, 32, image.size[0] - 32, image.size[1] - 32]])

    def predict(self, image: str | Path) -> OCRResult:
        results = self.predict_document(image)
        if not results:
            raise RuntimeError(f"OCR produced no results for input: {image}")
        return results[0]

    def predict_many(self, images: list[str | Path]) -> list[OCRResult]:
        results: list[OCRResult] = []
        for image in images:
            results.extend(self.predict_document(image))
        return results

    def predict_document(self, image: str | Path) -> list[OCRResult]:
        image_path = self._resolve_image_path(image)
        page_images = self._load_page_images(image_path)
        results: list[OCRResult] = []
        for page_index, page_image in enumerate(page_images):
            results.append(self._run_page(page_image, page_index=page_index, source_path=image_path))
        return results

    def build_document_payload(
        self,
        results: list[OCRResult],
        *,
        original_filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        pages = [result.to_dict() for result in results]
        raw_text = "\n\n".join(page["raw_text"] for page in pages if page["raw_text"].strip())
        full_text = "\n\n".join(page["full_text"] for page in pages if page["full_text"].strip())
        normalized_text = "\n\n".join(page["normalized_text"] for page in pages if page["normalized_text"].strip())
        markdown_text = self.build_document_markdown(
            results,
            original_filename=original_filename,
            content_type=content_type,
        )
        return {
            "raw_text": raw_text,
            "full_text": full_text,
            "normalized_text": normalized_text,
            "markdown_text": markdown_text,
            "pages": pages,
            "meta": {
                "page_count": len(results),
                "original_filename": original_filename,
                "content_type": content_type,
            },
        }

    def build_document_markdown(
        self,
        results: list[OCRResult],
        *,
        original_filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        del content_type
        pages: list[str] = []
        multi_page = len(results) > 1
        for index, result in enumerate(results, start=1):
            page_markdown = result.markdown_text.strip()
            if not page_markdown:
                page_markdown = self._text_fence("")
            if multi_page:
                pages.append(f"## Page {index}\n\n{page_markdown}")
            else:
                pages.append(page_markdown)
        markdown = "\n\n---\n\n".join(pages).strip()
        if original_filename:
            return f"<!-- source: {original_filename} -->\n\n{markdown}\n"
        return f"{markdown}\n"

    def _run_page(self, page_image: np.ndarray, *, page_index: int, source_path: Path) -> OCRResult:
        total_started = perf_counter()
        debug: dict[str, Any] = {}
        timings_ms: dict[str, float | None] = {}

        image = Image.fromarray(page_image).convert("RGB")

        started = perf_counter()
        image, polarity_debug = auto_invert_dark_background(image)
        timings_ms["polarity_preprocess"] = round((perf_counter() - started) * 1000.0, 2)
        debug["polarity_preprocess"] = polarity_debug

        working = image
        if self.config.use_doc_orientation_classify:
            started = perf_counter()
            try:
                ori_res = self.runtime.predict_doc_orientation(pil_to_numpy(working))
                orientation_label = self._extract_first_label(ori_res)
                working = rotate_by_label(working, orientation_label)
                debug["doc_orientation"] = {"label": orientation_label, "raw": self._sanitize(ori_res)}
            except Exception as exc:
                debug["doc_orientation"] = {"error": str(exc)}
            timings_ms["doc_orientation"] = round((perf_counter() - started) * 1000.0, 2)

        if self.config.use_doc_unwarping:
            started = perf_counter()
            try:
                unwarp_res = self.runtime.predict_doc_unwarp(pil_to_numpy(working))
                unwarp_img = unwarp_res.get("doctr_img")
                if isinstance(unwarp_img, np.ndarray) and unwarp_img.size:
                    working = Image.fromarray(unwarp_img.astype(np.uint8)).convert("RGB")
                debug["doc_unwarping"] = {"raw": self._sanitize(unwarp_res)}
            except Exception as exc:
                debug["doc_unwarping"] = {"error": str(exc)}
            timings_ms["doc_unwarping"] = round((perf_counter() - started) * 1000.0, 2)

        layout_blocks: list[dict[str, Any]] = []
        if self.config.use_layout_detection:
            started = perf_counter()
            try:
                layout_res = self.runtime.predict_layout(pil_to_numpy(working))
                layout_blocks = self._extract_layout_boxes(layout_res)
                debug["layout"] = self._sanitize(layout_res)
            except Exception as exc:
                debug["layout"] = {"error": str(exc)}
            timings_ms["layout"] = round((perf_counter() - started) * 1000.0, 2)

        region_blocks: list[dict[str, Any]] = []
        debug["regions"] = {"skipped": True, "reason": "region_detection_removed"}
        timings_ms["regions"] = None

        formula_items: list[dict[str, Any]] = []
        masked_for_text = working
        if self.config.use_formula_recognition:
            formula_boxes = [item["bbox"] for item in layout_blocks if "formula" in item.get("label", "").lower()]
            if formula_boxes:
                started = perf_counter()
                try:
                    formula_items = self._run_formula(working, formula_boxes)
                    debug["formula"] = {"count": len(formula_items)}
                    if formula_items:
                        masked_for_text = mask_bboxes(working, [item["bbox"] for item in formula_items])
                    else:
                        timings_ms["ocr_after_formula_mask"] = None
                except Exception as exc:
                    debug["formula"] = {"error": str(exc)}
                timings_ms["formula"] = round((perf_counter() - started) * 1000.0, 2)
            else:
                timings_ms["formula"] = None
                timings_ms["ocr_after_formula_mask"] = None
        else:
            timings_ms["formula"] = None
            timings_ms["ocr_after_formula_mask"] = None

        started = perf_counter()
        text_items = self._run_text(masked_for_text)
        timings_ms["ocr_primary"] = round((perf_counter() - started) * 1000.0, 2)
        debug["ocr"] = {"count": len(text_items)}

        merged_items = self._merge_items(text_items, formula_items, region_blocks)
        merged_items = self._filter_single_char_rows(merged_items)
        markdown = self._to_markdown(merged_items)
        lines = self._build_lines(merged_items, page_index=page_index)
        blocks = self._build_blocks(merged_items, page_index=page_index)
        regions = self._build_regions(region_blocks, page_index=page_index)
        formulas = self._build_formulas(formula_items, page_index=page_index)

        raw_text = "\n".join(line.text for line in lines if line.accepted and line.text.strip())
        total_elapsed = round((perf_counter() - total_started) * 1000.0, 2)
        timings_ms["total"] = total_elapsed

        return OCRResult(
            raw_text=raw_text,
            full_text=raw_text,
            normalized_text=raw_text,
            markdown_text=markdown,
            lines=lines,
            blocks=blocks,
            warnings=[],
            timings_ms=timings_ms,
            regions=regions,
            formulas=formulas,
            meta={
                "page_index": page_index,
                "page_number": page_index + 1,
                "source_path": str(source_path),
                "image_size": [int(working.size[0]), int(working.size[1])],
                "profile": self.config.profile,
                "device": self.config.device,
                "engine": self.config.engine,
                "use_document_structure": self.config.use_document_structure,
                "use_layout_detection": self.config.use_layout_detection,
                "use_region_detection": False,
                "use_formula_recognition": self.config.use_formula_recognition,
                "debug": debug,
            },
        )

    def _run_formula(self, image: Image.Image, bboxes: list[list[int]]) -> list[dict[str, Any]]:
        width, height = image.size
        clipped = [clip_bbox(box, width, height) for box in bboxes]
        crops = [pil_to_numpy(crop_by_bbox(image, box)) for box in clipped]
        raw_results = self.runtime.predict_formula(
            crops,
            batch_size=max(1, self.config.formula_recognition_batch_size),
        )
        items: list[dict[str, Any]] = []
        for idx, box in enumerate(clipped):
            raw = raw_results[idx] if idx < len(raw_results) else {}
            formula = str(raw.get("rec_formula") or raw.get("rec_text") or "").strip()
            if not formula:
                continue
            items.append(
                {
                    "type": "formula",
                    "bbox": box,
                    "text": f"\\[{ ' '.join(formula.splitlines()) }\\]",
                    "latex": formula,
                    "score": float(raw.get("rec_score", 1.0)),
                }
            )
        return items

    def _run_text(self, image: Image.Image) -> list[dict[str, Any]]:
        detect_res = self.runtime.predict_text_detection(pil_to_numpy(image))
        polys = detect_res.get("dt_polys")
        scores = detect_res.get("dt_scores")
        if isinstance(polys, np.ndarray):
            polys = polys.tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        if polys is None:
            polys = []
        if scores is None:
            scores = []

        width, height = image.size
        boxes: list[list[int]] = []
        det_scores: list[float] = []
        for idx, poly in enumerate(polys):
            bbox = clip_bbox(polygon_to_bbox(poly), width, height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            boxes.append(bbox)
            det_scores.append(float(scores[idx]) if idx < len(scores) else 0.0)

        if not boxes:
            return []

        crops = [pil_to_numpy(crop_by_bbox(image, box)) for box in boxes]
        if self.config.use_textline_orientation and crops:
            ori_results = self.runtime.predict_textline_orientation(crops)
            fixed_crops: list[np.ndarray] = []
            for idx, crop in enumerate(crops):
                raw = ori_results[idx] if idx < len(ori_results) else {}
                label = self._extract_first_label(raw)
                if label and "180" in str(label):
                    fixed_crops.append(np.rot90(crop, 2))
                else:
                    fixed_crops.append(crop)
            crops = fixed_crops

        rec_results = self.runtime.predict_text_recognition(
            crops,
            batch_size=max(1, self.config.text_recognition_batch_size),
        )

        items: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            rec = rec_results[idx] if idx < len(rec_results) else {}
            text = str(rec.get("rec_text", "")).strip()
            if not text:
                continue
            items.append(
                {
                    "type": "text",
                    "bbox": box,
                    "polygon": bbox_to_polygon(box),
                    "text": text,
                    "score": float(rec.get("rec_score", det_scores[idx])),
                }
            )
        return items

    def _merge_items(
        self,
        text_items: list[dict[str, Any]],
        formula_items: list[dict[str, Any]],
        regions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = sort_reading_order(text_items + formula_items)
        for item in merged:
            item["region_id"] = self._find_region_id(item["bbox"], regions) if regions else None
        return merged

    @staticmethod
    def _is_drop_single_char_text(text: str) -> bool:
        normalized = text.replace("\t", "").replace(" ", "")
        if len(normalized) != 1:
            return False
        return normalized not in _SINGLE_CHAR_ALLOWED

    def _filter_single_char_rows(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for item in items:
            if str(item.get("type") or "") != "text":
                filtered.append(item)
                continue
            text = str(item.get("text") or "")
            if self._is_drop_single_char_text(text):
                continue
            filtered.append(item)
        return filtered

    def _build_lines(self, items: list[dict[str, Any]], *, page_index: int) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for order, item in enumerate(items, start=1):
            bbox = list(item.get("bbox") or [0, 0, 0, 0])
            polygon = item.get("polygon") or bbox_to_polygon(bbox)
            score = item.get("score")
            lines.append(
                OCRLine(
                    order=order,
                    text=str(item.get("text") or ""),
                    normalized_text=str(item.get("text") or ""),
                    det_score=float(score) if score is not None else None,
                    rec_score=float(score) if score is not None else None,
                    polygon=polygon,
                    bbox=bbox,
                    page_index=page_index,
                    accepted=True,
                    flags=[str(item.get("type") or "text")],
                )
            )
        return lines

    def _build_blocks(self, items: list[dict[str, Any]], *, page_index: int) -> list[OCRBlock]:
        blocks: list[OCRBlock] = []
        for order, item in enumerate(items, start=1):
            text = str(item.get("text") or "")
            blocks.append(
                OCRBlock(
                    id=f"p{page_index + 1}_b{order}",
                    order=order,
                    kind=str(item.get("type") or "text"),
                    text=text,
                    normalized_text=text,
                    page_index=page_index,
                    line_orders=[order],
                    bbox=list(item.get("bbox") or [0, 0, 0, 0]),
                    confidence=float(item["score"]) if item.get("score") is not None else None,
                    cells=[],
                )
            )
        return blocks

    def _build_regions(self, items: list[dict[str, Any]], *, page_index: int) -> list[OCRRegion]:
        regions: list[OCRRegion] = []
        for order, item in enumerate(items, start=1):
            regions.append(
                OCRRegion(
                    id=f"p{page_index + 1}_r{order}",
                    order=order,
                    label=str(item.get("label") or ""),
                    bbox=list(item.get("bbox") or [0, 0, 0, 0]),
                    page_index=page_index,
                    confidence=float(item["score"]) if item.get("score") is not None else None,
                    source="layout",
                )
            )
        return regions

    def _build_formulas(self, items: list[dict[str, Any]], *, page_index: int) -> list[OCRFormula]:
        formulas: list[OCRFormula] = []
        for order, item in enumerate(items, start=1):
            formulas.append(
                OCRFormula(
                    id=f"p{page_index + 1}_f{order}",
                    order=order,
                    latex=str(item.get("latex") or ""),
                    page_index=page_index,
                    bbox=list(item.get("bbox") or [0, 0, 0, 0]),
                    confidence=float(item["score"]) if item.get("score") is not None else None,
                    line_orders=[],
                    source="formula_recognition",
                )
            )
        return formulas

    @staticmethod
    def _to_markdown(items: list[dict[str, Any]]) -> str:
        lines = [str(item.get("text") or "").strip() for item in items if str(item.get("text") or "").strip()]
        return "\n".join(lines)

    @staticmethod
    def _text_fence(text: str) -> str:
        return f"```\n{text.rstrip()}\n```"

    @staticmethod
    def _extract_first_label(raw: dict[str, Any]) -> str | None:
        labels = raw.get("label_names")
        if isinstance(labels, list) and labels:
            return str(labels[0])
        if isinstance(labels, np.ndarray) and labels.size > 0:
            return str(labels.tolist()[0])
        return None

    @staticmethod
    def _extract_layout_boxes(raw: dict[str, Any]) -> list[dict[str, Any]]:
        boxes = raw.get("boxes") or []
        normalized: list[dict[str, Any]] = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            coordinate = box.get("coordinate") or [0, 0, 0, 0]
            if len(coordinate) < 4:
                continue
            normalized.append(
                {
                    "bbox": [int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[3])],
                    "label": str(box.get("label", "")),
                    "score": float(box.get("score", 0.0)),
                }
            )
        return normalized

    @staticmethod
    def _find_region_id(bbox: list[int], regions: list[dict[str, Any]]) -> int | None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        for idx, region in enumerate(regions):
            rx1, ry1, rx2, ry2 = region["bbox"]
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return idx
        return None

    @staticmethod
    def _rotate_bbox(bbox: list[int], label: str, width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = bbox
        if label == "90":
            points = [(height - y2, x1), (height - y1, x2)]
        elif label == "180":
            points = [(width - x2, height - y2), (width - x1, height - y1)]
        elif label == "270":
            points = [(y1, width - x2), (y2, width - x1)]
        else:
            return bbox
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    def _rotate_items(self, items: list[dict[str, Any]], label: str, size: tuple[int, int]) -> list[dict[str, Any]]:
        normalized = label.replace("°", "").strip()
        if normalized not in {"90", "180", "270"}:
            return items
        width, height = size
        rotated: list[dict[str, Any]] = []
        for item in items:
            updated = dict(item)
            bbox = list(item.get("bbox") or [0, 0, 0, 0])
            updated_bbox = self._rotate_bbox(bbox, normalized, width, height)
            updated["bbox"] = updated_bbox
            updated["polygon"] = bbox_to_polygon(updated_bbox)
            rotated.append(updated)
        return rotated

    def _resolve_image_path(self, image: str | Path) -> Path:
        image_path = Path(image).expanduser()
        if not image_path.is_absolute():
            image_path = (Path.cwd() / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image path is not a file: {image_path}")
        return image_path

    def _load_page_images(self, image_path: Path) -> list[np.ndarray]:
        if image_path.suffix.lower() != ".pdf":
            return [self._load_image_rgb(image_path)]

        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("pypdfium2 is required to rasterize PDF uploads.") from exc

        document = pdfium.PdfDocument(str(image_path))
        images: list[np.ndarray] = []
        try:
            for index in range(len(document)):
                page = document[index]
                bitmap = page.render(scale=2.0).to_pil().convert("RGB")
                images.append(np.asarray(bitmap))
        finally:
            document.close()
        return images

    @staticmethod
    def _load_image_rgb(image_path: Path) -> np.ndarray:
        return np.asarray(Image.open(image_path).convert("RGB"))

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, np.ndarray):
            if value.ndim >= 2:
                return {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value
