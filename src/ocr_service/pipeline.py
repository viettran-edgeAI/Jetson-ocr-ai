from __future__ import annotations

import importlib.util
import math
import os
import platform
import re
import sys
import warnings
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any


def _default_model_dir(name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "models" / name


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _env_str(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            return raw
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return values or default


def _normalize_profile(profile: str | None) -> str:
    normalized = (profile or "").strip().lower()
    if normalized in {"full", "quality"}:
        return "full"
    if normalized in {"fast", "speed", ""}:
        return "fast"
    return "fast"


@dataclass(slots=True)
class OCRLine:
    order: int
    text: str
    normalized_text: str = ""
    det_score: float | None = None
    rec_score: float | None = None
    polygon: list[list[int]] | None = None
    bbox: list[int] | None = None
    page_index: int | None = None
    accepted: bool = True
    flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OCRBlock:
    id: str
    order: int
    kind: str
    text: str
    normalized_text: str
    page_index: int
    line_orders: list[int]
    bbox: list[int] | None = None
    confidence: float | None = None
    cells: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OCRResult:
    raw_text: str
    full_text: str
    normalized_text: str
    markdown_text: str
    lines: list[OCRLine]
    blocks: list[OCRBlock]
    warnings: list[dict[str, Any]]
    timings_ms: dict[str, float | None]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "full_text": self.full_text,
            "normalized_text": self.normalized_text,
            "markdown_text": self.markdown_text,
            "lines": [
                {
                    "order": line.order,
                    "text": line.text,
                    "normalized_text": line.normalized_text,
                    "det_score": line.det_score,
                    "rec_score": line.rec_score,
                    "polygon": line.polygon,
                    "bbox": line.bbox,
                    "page_index": line.page_index,
                    "accepted": line.accepted,
                    "flags": list(line.flags),
                }
                for line in self.lines
            ],
            "blocks": [
                {
                    "id": block.id,
                    "order": block.order,
                    "kind": block.kind,
                    "text": block.text,
                    "normalized_text": block.normalized_text,
                    "page_index": block.page_index,
                    "line_orders": list(block.line_orders),
                    "bbox": block.bbox,
                    "confidence": block.confidence,
                    "cells": list(block.cells),
                }
                for block in self.blocks
            ],
            "warnings": [dict(warning) for warning in self.warnings],
            "timings_ms": dict(self.timings_ms),
            "meta": dict(self.meta),
        }


class OCRPipeline:
    def __init__(
        self,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        device: str | None = None,
        doc_orientation_model_dir: str | None = None,
        doc_unwarping_model_dir: str | None = None,
        textline_orientation_model_dir: str | None = None,
        profile: str | None = None,
        engine: str | None = None,
        text_recognition_batch_size: int | None = None,
        textline_orientation_batch_size: int | None = None,
        enable_hpi: bool | None = None,
        use_tensorrt: bool | None = None,
        trt_profile: str | None = None,
        trt_modules: tuple[str, ...] | None = None,
    ) -> None:
        requested_tensorrt = use_tensorrt if use_tensorrt is not None else _env_bool("OCR_USE_TENSORRT", False)
        if requested_tensorrt:
            self._prepare_tensorrt_process_flags()

        try:
            import paddle
        except ImportError:
            paddle = None

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - runtime dependency check
            raise RuntimeError(
                "paddleocr is not installed. Install Jetson-compatible PaddleOCR and PaddlePaddle first."
            ) from exc

        self.doc_orientation_model_name = "PP-LCNet_x1_0_doc_ori"
        self.doc_unwarping_model_name = "UVDoc"
        self.textline_orientation_model_name = "PP-LCNet_x0_25_textline_ori"
        self.det_model_name = "PP-OCRv5_mobile_det"
        self.rec_model_name = "PP-OCRv5_mobile_rec"

        self.doc_orientation_model_dir = Path(
            doc_orientation_model_dir
            or os.environ.get("OCR_DOC_ORI_MODEL_DIR")
            or _default_model_dir("PP-LCNet_x1_0_doc_ori_infer")
        )
        self.doc_unwarping_model_dir = Path(
            doc_unwarping_model_dir
            or os.environ.get("OCR_DOC_UNWARP_MODEL_DIR")
            or _default_model_dir("UVDoc_infer")
        )
        self.textline_orientation_model_dir = Path(
            textline_orientation_model_dir
            or os.environ.get("OCR_TEXTLINE_ORI_MODEL_DIR")
            or _default_model_dir("PP-LCNet_x0_25_textline_ori_infer")
        )
        self.det_model_dir = Path(
            det_model_dir or os.environ.get("OCR_DET_MODEL_DIR") or _default_model_dir(self.det_model_name)
        )
        self.rec_model_dir = Path(
            rec_model_dir or os.environ.get("OCR_REC_MODEL_DIR") or _default_model_dir(self.rec_model_name)
        )

        self.profile = _normalize_profile(profile or os.environ.get("OCR_PROFILE"))
        default_optional_modules = self.profile == "full"

        self.use_doc_orientation_classify = _env_bool(
            "OCR_USE_DOC_ORIENTATION_CLASSIFY",
            default_optional_modules,
        )
        self.use_doc_unwarping = _env_bool(
            "OCR_USE_DOC_UNWARPING",
            default_optional_modules,
        )
        self.use_textline_orientation = _env_bool(
            "OCR_USE_TEXTLINE_ORIENTATION",
            default_optional_modules,
        )
        self.text_det_limit_side_len = _env_int(("OCR_TEXT_DET_LIMIT_SIDE_LEN", "OCR_DET_LIMIT_SIDE_LEN"), 64)
        self.text_det_limit_type = _env_str(("OCR_TEXT_DET_LIMIT_TYPE", "OCR_DET_LIMIT_TYPE"), "min")
        self.text_recognition_batch_size = text_recognition_batch_size or _env_int(
            ("OCR_TEXT_RECOGNITION_BATCH_SIZE", "OCR_REC_BATCH_SIZE"),
            4,
        )
        self.textline_orientation_batch_size = textline_orientation_batch_size or _env_int(
            ("OCR_TEXTLINE_ORI_BATCH_SIZE",),
            6,
        )
        self.engine = engine or os.environ.get("OCR_ENGINE") or "paddle_static"
        self.enable_hpi = enable_hpi if enable_hpi is not None else _env_bool("OCR_ENABLE_HPI", False)
        self.use_tensorrt = requested_tensorrt
        self.run_mode = _env_str(("OCR_RUN_MODE",), "trt_fp16" if self.use_tensorrt else "paddle")
        self.trt_profile = (trt_profile or os.environ.get("OCR_TRT_PROFILE") or "jetson").strip().lower()
        self.trt_modules = trt_modules or _env_list("OCR_TRT_MODULES", ("det", "rec"))
        self.trt_workspace_mb = _env_int(("OCR_TRT_WORKSPACE_MB",), 384)
        self.trt_det_max_side = _env_int(("OCR_TRT_DET_MAX_SIDE",), 960)
        self.trt_det_opt_side = _env_int(("OCR_TRT_DET_OPT_SIDE",), 256)
        self.trt_rec_max_width = _env_int(("OCR_TRT_REC_MAX_WIDTH",), 1600)

        requested_device = device or os.environ.get("OCR_DEVICE")
        if requested_device:
            self.device = self._resolve_device(requested_device, paddle)
        else:
            self.device = self._default_device(paddle)

        self._prepare_acceleration_flags()
        engine_config = self._build_engine_config()
        paddlex_config = self._build_paddlex_config()

        self._ocr = PaddleOCR(
            paddlex_config=paddlex_config,
            doc_orientation_classify_model_name=self.doc_orientation_model_name,
            doc_orientation_classify_model_dir=str(self.doc_orientation_model_dir),
            doc_unwarping_model_name=self.doc_unwarping_model_name,
            doc_unwarping_model_dir=str(self.doc_unwarping_model_dir),
            text_detection_model_name=self.det_model_name,
            text_detection_model_dir=str(self.det_model_dir),
            textline_orientation_model_name=self.textline_orientation_model_name,
            textline_orientation_model_dir=str(self.textline_orientation_model_dir),
            text_recognition_model_name=self.rec_model_name,
            text_recognition_model_dir=str(self.rec_model_dir),
            textline_orientation_batch_size=self.textline_orientation_batch_size,
            text_recognition_batch_size=self.text_recognition_batch_size,
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            use_textline_orientation=self.use_textline_orientation,
            text_det_limit_side_len=self.text_det_limit_side_len,
            text_det_limit_type=self.text_det_limit_type,
            device=self.device,
            engine=self.engine,
            enable_hpi=self.enable_hpi,
            engine_config=engine_config,
        )

    def _prepare_tensorrt_process_flags(self) -> None:
        tensorrt_python_path = os.environ.get("OCR_TENSORRT_PYTHON_PATH")
        default_tensorrt_python_path = "/usr/lib/python3.10/dist-packages"
        if not tensorrt_python_path and os.path.isdir(default_tensorrt_python_path):
            tensorrt_python_path = default_tensorrt_python_path

        if tensorrt_python_path and tensorrt_python_path not in sys.path:
            sys.path.append(tensorrt_python_path)

        if "PADDLE_PDX_USE_PIR_TRT" not in os.environ:
            # The PaddleX PIR TensorRT exporter currently fails on this Jetson stack with
            # TensorRT layer-name API errors. The legacy Paddle-TRT subgraph path is the
            # more conservative runtime for Jetson validation.
            os.environ["PADDLE_PDX_USE_PIR_TRT"] = "0"

        # PaddleX reads this flag at import time. If a process has already created
        # a plain Paddle pipeline before enabling TRT, update the loaded modules too.
        for module_name in (
            "paddlex.utils.flags",
            "paddlex.inference.models.runners.paddle_static.config.trt_config",
            "paddlex.inference.models.runners.paddle_static.config.pp_option",
            "paddlex.inference.models.runners.paddle_static.runner",
        ):
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "USE_PIR_TRT"):
                setattr(module, "USE_PIR_TRT", False)

        trt_config_module = sys.modules.get(
            "paddlex.inference.models.runners.paddle_static.config.trt_config"
        )
        if trt_config_module is not None and hasattr(trt_config_module, "OLD_IR_TRT_PRECISION_MAP"):
            trt_config_module.TRT_PRECISION_MAP = trt_config_module.OLD_IR_TRT_PRECISION_MAP

        pp_option_module = sys.modules.get(
            "paddlex.inference.models.runners.paddle_static.config.pp_option"
        )
        if pp_option_module is not None and trt_config_module is not None:
            pp_option_module.TRT_PRECISION_MAP = trt_config_module.OLD_IR_TRT_PRECISION_MAP

    def _default_device(self, paddle_module: Any | None) -> str:
        if paddle_module is not None and getattr(paddle_module, "is_compiled_with_cuda", lambda: False)():
            return "gpu:0"
        return "cpu"

    def _resolve_device(self, requested_device: str, paddle_module: Any | None) -> str:
        if requested_device.startswith("gpu"):
            has_cuda = bool(
                paddle_module is not None and getattr(paddle_module, "is_compiled_with_cuda", lambda: False)()
            )
            if not has_cuda:
                warnings.warn(
                    "OCR_DEVICE requested GPU, but the installed PaddlePaddle runtime does not have CUDA support; falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return "cpu"
        return requested_device

    def _prepare_acceleration_flags(self) -> None:
        if self.enable_hpi and platform.machine().lower() not in {"x86_64", "amd64"}:
            warnings.warn(
                "OCR_ENABLE_HPI was requested, but PaddleOCR documents high-performance inference for x86-64 only; disabling it on this Jetson ARM64 runtime.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.enable_hpi = False

        if not self.use_tensorrt:
            return

        if self.device == "cpu":
            warnings.warn(
                "OCR_USE_TENSORRT was requested while OCR is running on CPU; disabling TensorRT.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.use_tensorrt = False
            return

        tensorrt_python_path = os.environ.get("OCR_TENSORRT_PYTHON_PATH")
        default_tensorrt_python_path = "/usr/lib/python3.10/dist-packages"
        if not tensorrt_python_path and os.path.isdir(default_tensorrt_python_path):
            tensorrt_python_path = default_tensorrt_python_path

        if tensorrt_python_path and tensorrt_python_path not in sys.path:
            # Append the system TensorRT bindings without overriding the venv's NumPy/OpenCV packages.
            sys.path.append(tensorrt_python_path)

        if importlib.util.find_spec("tensorrt") is None:
            warnings.warn(
                "OCR_USE_TENSORRT was requested, but the TensorRT Python bindings are not importable; falling back to plain paddle_static.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.use_tensorrt = False
            return

        self._patch_paddlex_tensorrt_defaults()

    def _patch_paddlex_tensorrt_defaults(self) -> None:
        try:
            from paddle.inference import PrecisionType
            from paddlex.inference.models.runners.paddle_static.config import trt_config
        except Exception as exc:
            warnings.warn(
                f"TensorRT was requested, but PaddleX TensorRT defaults could not be patched: {exc!r}; falling back to plain paddle_static.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.use_tensorrt = False
            return

        workspace_bytes = max(64, self.trt_workspace_mb) << 20
        precision = PrecisionType.Half if self.run_mode == "trt_fp16" else PrecisionType.Float32
        setting = {
            "enable_tensorrt_engine": {
                "workspace_size": workspace_bytes,
                "max_batch_size": max(1, self.text_recognition_batch_size),
                "min_subgraph_size": 3,
                "precision_mode": precision,
                "use_static": True,
                "use_calib_mode": False,
            }
        }

        for model_name in (
            self.det_model_name,
            self.rec_model_name,
            self.textline_orientation_model_name,
            self.doc_orientation_model_name,
            self.doc_unwarping_model_name,
        ):
            trt_config.TRT_CFG_SETTING[model_name] = deepcopy(setting)

    def _build_engine_config(self) -> dict[str, Any] | None:
        if not self.use_tensorrt:
            return None
        return {"run_mode": "paddle"}

    def _build_paddlex_config(self) -> dict[str, Any] | None:
        if not self.use_tensorrt:
            return None

        try:
            from paddlex.inference import load_pipeline_config
        except Exception as exc:
            warnings.warn(
                f"TensorRT was requested, but the PaddleX OCR config could not be loaded: {exc!r}; falling back to plain paddle_static.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.use_tensorrt = False
            return None

        config = self._to_builtin(load_pipeline_config("OCR"))
        modules = set(self.trt_modules)
        if "all" in modules:
            modules = {"det", "rec", "textline", "doc_ori", "doc_unwarp"}

        det_config = self._trt_det_engine_config() if "det" in modules else {"run_mode": "paddle"}
        rec_config = self._trt_rec_engine_config() if "rec" in modules else {"run_mode": "paddle"}

        config.setdefault("SubModules", {}).setdefault("TextDetection", {})["engine_config"] = det_config
        config.setdefault("SubModules", {}).setdefault("TextRecognition", {})["engine_config"] = rec_config
        config["SubModules"].setdefault("TextLineOrientation", {})["engine_config"] = (
            self._trt_textline_engine_config() if "textline" in modules else {"run_mode": "paddle"}
        )

        doc_preprocessor = config.setdefault("SubPipelines", {}).setdefault("DocPreprocessor", {})
        doc_modules = doc_preprocessor.setdefault("SubModules", {})
        doc_modules.setdefault("DocOrientationClassify", {})["engine_config"] = (
            self._trt_doc_ori_engine_config() if "doc_ori" in modules else {"run_mode": "paddle"}
        )
        doc_modules.setdefault("DocUnwarping", {})["engine_config"] = (
            self._trt_doc_unwarp_engine_config() if "doc_unwarp" in modules else {"run_mode": "paddle"}
        )

        return config

    def _trt_det_engine_config(self) -> dict[str, Any]:
        min_side = 32
        opt_side = max(64, min(self.trt_det_opt_side, self.trt_det_max_side))
        max_side = max(opt_side, self.trt_det_max_side)
        return {
            "run_mode": self.run_mode,
            "trt_dynamic_shapes": {
                "x": [
                    [1, 3, min_side, min_side],
                    [1, 3, opt_side, opt_side],
                    [1, 3, max_side, max_side],
                ]
            },
            "trt_collect_shape_range_info": False,
            "trt_allow_rebuild_at_runtime": True,
        }

    def _trt_rec_engine_config(self) -> dict[str, Any]:
        batch = max(1, min(self.text_recognition_batch_size, _env_int(("OCR_TRT_REC_MAX_BATCH",), 4)))
        max_width = max(320, self.trt_rec_max_width)
        opt_width = min(max_width, _env_int(("OCR_TRT_REC_OPT_WIDTH",), 480))
        return {
            "run_mode": self.run_mode,
            "trt_dynamic_shapes": {
                "x": [
                    [1, 3, 48, 160],
                    [batch, 3, 48, opt_width],
                    [batch, 3, 48, max_width],
                ]
            },
            "trt_collect_shape_range_info": False,
            "trt_allow_rebuild_at_runtime": True,
        }

    def _trt_textline_engine_config(self) -> dict[str, Any]:
        batch = max(1, min(self.textline_orientation_batch_size, _env_int(("OCR_TRT_TEXTLINE_MAX_BATCH",), 6)))
        return {
            "run_mode": self.run_mode,
            "trt_dynamic_shapes": {"x": [[1, 3, 48, 96], [batch, 3, 48, 192], [batch, 3, 48, 320]]},
            "trt_collect_shape_range_info": False,
            "trt_allow_rebuild_at_runtime": True,
        }

    def _trt_doc_ori_engine_config(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "trt_dynamic_shapes": {"x": [[1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224]]},
            "trt_collect_shape_range_info": False,
            "trt_allow_rebuild_at_runtime": True,
        }

    def _trt_doc_unwarp_engine_config(self) -> dict[str, Any]:
        side = max(320, _env_int(("OCR_TRT_DOC_UNWARP_MAX_SIDE",), 960))
        return {
            "run_mode": self.run_mode,
            "trt_dynamic_shapes": {"x": [[1, 3, 320, 320], [1, 3, 736, 736], [1, 3, side, side]]},
            "trt_collect_shape_range_info": False,
            "trt_allow_rebuild_at_runtime": True,
        }

    def _to_builtin(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._to_builtin(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_builtin(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_builtin(item) for item in value]
        return value

    def predict(self, image: str | Path) -> OCRResult:
        results = self.predict_document(image)
        if not results:
            raise RuntimeError(f"OCR produced no results for input: {image}")
        return results[0]

    def predict_document(self, image: str | Path) -> list[OCRResult]:
        start_total = perf_counter()

        start_preprocess = perf_counter()
        image_path = self._resolve_image_path(image)
        image_size = self._get_image_size(image_path)
        preprocess_ms = (perf_counter() - start_preprocess) * 1000.0

        start_pipeline = perf_counter()
        prediction = self._ocr.predict(str(image_path))
        pipeline_ms = (perf_counter() - start_pipeline) * 1000.0

        start_postprocess = perf_counter()
        payloads = self._extract_payloads(prediction)
        results: list[OCRResult] = []
        for payload_index, payload in enumerate(payloads):
            page_index = self._coerce_page_index(payload.get("page_index"), payload_index)
            lines = self._build_lines(payload, page_index=page_index)
            line_warnings = self._apply_line_filters(lines)
            blocks = self._build_blocks(lines, page_index=page_index)
            page_warnings = line_warnings + self._build_page_warnings(lines=lines, page_index=page_index)
            raw_text = "\n".join(line.text for line in lines if line.accepted and line.text.strip())
            full_text = self._blocks_to_full_text(blocks) or raw_text
            normalized_text = self._normalize_text(full_text, preserve_newlines=True)
            markdown_text = self._page_to_markdown(blocks, fallback_text=raw_text)
            doc_preprocessor_res = payload.get("doc_preprocessor_res")
            results.append(
                OCRResult(
                    raw_text=raw_text,
                    full_text=full_text,
                    normalized_text=normalized_text,
                    markdown_text=markdown_text,
                    lines=lines,
                    blocks=blocks,
                    warnings=page_warnings,
                    timings_ms={},
                    meta={
                        "profile": self.profile,
                        "device": self.device,
                        "engine": self.engine,
                        "enable_hpi": self.enable_hpi,
                        "use_tensorrt": self.use_tensorrt,
                        "run_mode": self.run_mode if self.use_tensorrt else None,
                        "trt_profile": self.trt_profile if self.use_tensorrt else None,
                        "trt_modules": list(self.trt_modules) if self.use_tensorrt else None,
                        "trt_workspace_mb": self.trt_workspace_mb if self.use_tensorrt else None,
                        "image_size": list(image_size) if image_size else None,
                        "doc_orientation_classify_model": self.doc_orientation_model_name,
                        "doc_orientation_classify_model_dir": str(self.doc_orientation_model_dir),
                        "doc_unwarping_model": self.doc_unwarping_model_name,
                        "doc_unwarping_model_dir": str(self.doc_unwarping_model_dir),
                        "textline_orientation_model": self.textline_orientation_model_name,
                        "textline_orientation_model_dir": str(self.textline_orientation_model_dir),
                        "det_model": self.det_model_name,
                        "det_model_dir": str(self.det_model_dir),
                        "rec_model": self.rec_model_name,
                        "rec_model_dir": str(self.rec_model_dir),
                        "use_doc_orientation_classify": self.use_doc_orientation_classify,
                        "use_doc_unwarping": self.use_doc_unwarping,
                        "use_textline_orientation": self.use_textline_orientation,
                        "textline_orientation_batch_size": self.textline_orientation_batch_size,
                        "text_recognition_batch_size": self.text_recognition_batch_size,
                        "text_det_limit_side_len": self.text_det_limit_side_len,
                        "text_det_limit_type": self.text_det_limit_type,
                        "model_settings": payload.get("model_settings"),
                        "doc_preprocessor_res": doc_preprocessor_res,
                        "text_det_params": payload.get("text_det_params"),
                        "page_index": page_index,
                        "page_number": page_index + 1,
                        "line_count": len(lines),
                        "accepted_line_count": len([line for line in lines if line.accepted and line.normalized_text]),
                        "layout_row_count": len(blocks),
                    },
                )
            )
        postprocess_ms = (perf_counter() - start_postprocess) * 1000.0

        total_ms = (perf_counter() - start_total) * 1000.0
        timings = {
            "preprocess": round(preprocess_ms, 2),
            "pipeline": round(pipeline_ms, 2),
            "postprocess": round(postprocess_ms, 2),
            "total": round(total_ms, 2),
        }
        for result in results:
            result.timings_ms = dict(timings)

        return results

    def predict_many(self, images: list[str | Path]) -> list[OCRResult]:
        results: list[OCRResult] = []
        for image in images:
            results.extend(self.predict_document(image))
        return results

    def build_document_payload(
        self,
        results: list[OCRResult],
        *,
        original_filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        pages = [result.to_dict() for result in results]
        all_lines = [line for page in pages for line in page["lines"]]
        all_blocks = [block for page in pages for block in page["blocks"]]
        all_warnings = [warning for page in pages for warning in page["warnings"]]
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
            "lines": all_lines,
            "blocks": all_blocks,
            "warnings": all_warnings,
            "timings_ms": {
                "document_total": round(
                    sum(float(result.timings_ms.get("total") or 0.0) for result in results),
                    2,
                ),
                "page_count": len(results),
            },
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

    def _resolve_image_path(self, image: str | Path) -> Path:
        image_path = Path(image).expanduser()
        if not image_path.is_absolute():
            image_path = (Path.cwd() / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image path is not a file: {image_path}")
        return image_path

    def _get_image_size(self, image_path: Path) -> tuple[int, int] | None:
        if image_path.suffix.lower() == ".pdf":
            return None
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - runtime dependency check
            raise RuntimeError("Pillow is required to inspect image metadata for OCR results.") from exc

        with Image.open(image_path) as image:
            return image.size

    def _extract_payloads(self, prediction: Any) -> list[dict[str, Any]]:
        if not prediction:
            return []

        items = prediction if isinstance(prediction, (list, tuple)) else [prediction]
        payloads: list[dict[str, Any]] = []
        for item in items:
            payload = getattr(item, "json", None)
            if callable(payload):
                payload = payload()

            if isinstance(payload, dict):
                payloads.append(payload.get("res", payload))
            elif isinstance(item, dict):
                payloads.append(item.get("res", item))
            else:
                payloads.append({"raw": item})
        return payloads

    def _build_lines(self, payload: dict[str, Any], *, page_index: int) -> list[OCRLine]:
        rec_texts = self._coerce_list(payload.get("rec_texts"))
        rec_scores = self._coerce_list(payload.get("rec_scores"))
        det_scores = self._coerce_list(payload.get("dt_scores"))
        polygons = self._coerce_list(payload.get("rec_polys") or payload.get("dt_polys"))

        lines: list[OCRLine] = []
        for index, text in enumerate(rec_texts, start=1):
            polygon = self._polygon_to_ints(self._list_item(polygons, index - 1))
            lines.append(
                OCRLine(
                    order=index,
                    text=str(text).strip(),
                    normalized_text=self._normalize_text(str(text).strip()),
                    det_score=self._maybe_float(det_scores, index - 1),
                    rec_score=self._maybe_float(rec_scores, index - 1),
                    polygon=polygon,
                    bbox=self._polygon_to_bbox(polygon),
                    page_index=page_index,
                )
            )

        if not lines:
            for index, raw_polygon in enumerate(polygons, start=1):
                polygon = self._polygon_to_ints(raw_polygon)
                lines.append(
                    OCRLine(
                        order=index,
                        text="",
                        normalized_text="",
                        det_score=self._maybe_float(det_scores, index - 1),
                        rec_score=None,
                        polygon=polygon,
                        bbox=self._polygon_to_bbox(polygon),
                        page_index=page_index,
                    )
                )

        lines = sorted(lines, key=self._line_sort_key)
        for order, line in enumerate(lines, start=1):
            line.order = order
        return lines

    def _apply_line_filters(self, lines: list[OCRLine]) -> list[dict[str, Any]]:
        warnings_out: list[dict[str, Any]] = []
        kept_lines: list[OCRLine] = []

        for line in lines:
            line.flags = []
            if not line.normalized_text:
                line.accepted = False
                line.flags.append("empty_text")
                continue

            if self._is_low_confidence(line):
                line.flags.append("low_confidence")
                warnings_out.append(
                    self._warning(
                        code="low_confidence_line",
                        severity="medium",
                        message=f"Line {line.order} has low recognition confidence.",
                        page_index=line.page_index,
                        line_orders=[line.order],
                    )
                )

            if self._is_short_marker_line(line):
                line.flags.append("standalone_marker")
                warnings_out.append(
                    self._warning(
                        code="standalone_marker_line",
                        severity="low",
                        message=f"Line {line.order} looks like a checkbox or marker and may need nearby text to interpret.",
                        page_index=line.page_index,
                        line_orders=[line.order],
                    )
                )

            duplicate = next(
                (candidate for candidate in reversed(kept_lines[-3:]) if self._is_duplicate_line(line, candidate)),
                None,
            )
            if duplicate is not None:
                line.accepted = False
                line.flags.append("duplicate_suppressed")
                warnings_out.append(
                    self._warning(
                        code="duplicate_line",
                        severity="low",
                        message=f"Line {line.order} was suppressed as a likely duplicate of line {duplicate.order}.",
                        page_index=line.page_index,
                        line_orders=[duplicate.order, line.order],
                    )
                )
                continue

            kept_lines.append(line)

        return warnings_out

    def _build_blocks(self, lines: list[OCRLine], *, page_index: int) -> list[OCRBlock]:
        rows = self._build_rows(lines)
        if not rows:
            return []

        blocks: list[OCRBlock] = []

        for row in rows:
            blocks.append(
                OCRBlock(
                    id=f"p{page_index + 1}_b{len(blocks) + 1}",
                    order=len(blocks) + 1,
                    kind="layout_row",
                    text=row["text"],
                    normalized_text=self._normalize_text(row["text"]),
                    page_index=page_index,
                    line_orders=list(row["line_orders"]),
                    bbox=row["bbox"],
                    confidence=self._mean_score(row["rec_scores"]),
                    cells=list(row["cells"]),
                )
            )

        return blocks

    def _build_rows(self, lines: list[OCRLine]) -> list[dict[str, Any]]:
        accepted_lines = [line for line in lines if line.accepted and line.normalized_text and line.bbox]
        if not accepted_lines:
            return []

        median_height = self._median_line_height(accepted_lines)
        skew_angle = self._estimate_page_skew_angle(accepted_lines)
        line_positions: list[dict[str, Any]] = []
        for line in accepted_lines:
            anchor_x, anchor_y = self._line_anchor_point(line)
            line_positions.append(
                {
                    "line": line,
                    "row_y": self._project_y(anchor_x, anchor_y, skew_angle),
                    "row_x": self._project_x(anchor_x, anchor_y, skew_angle),
                    "bbox": line.bbox,
                }
            )

        line_positions.sort(key=lambda item: (item["row_y"], item["row_x"], item["line"].order))

        raw_rows: list[dict[str, Any]] = []
        tolerance = max(8.0, min(24.0, median_height * 0.3))
        for item in line_positions:
            line = item["line"]
            row_y = item["row_y"]
            row_x = item["row_x"]
            if not raw_rows or abs(row_y - raw_rows[-1]["row_y"]) > tolerance:
                raw_rows.append({"lines": [line], "row_y": row_y, "row_x": row_x, "bbox": line.bbox})
                continue

            raw_rows[-1]["lines"].append(line)
            raw_rows[-1]["bbox"] = self._merge_bboxes([raw_rows[-1]["bbox"], line.bbox])
            raw_rows[-1]["row_y"] = self._mean_value(
                self._project_y(*self._line_anchor_point(item_line), skew_angle)
                for item_line in raw_rows[-1]["lines"]
                if item_line.bbox
            )
            raw_rows[-1]["row_x"] = self._mean_value(
                self._project_x(*self._line_anchor_point(item_line), skew_angle)
                for item_line in raw_rows[-1]["lines"]
                if item_line.bbox
            )

        rows: list[dict[str, Any]] = []
        page_left = min(self._line_anchor_point(line)[0] for line in accepted_lines if line.bbox)
        for row_index, row in enumerate(raw_rows, start=1):
            row_lines = sorted(
                row["lines"],
                key=lambda item: (self._project_x(*self._line_anchor_point(item), skew_angle), item.order),
            )
            row_text = self._join_row_text(row_lines, page_left=page_left)
            rows.append(
                {
                    "order": row_index,
                    "text": row_text,
                    "cells": [line.normalized_text for line in row_lines if line.normalized_text],
                    "kind": "layout_row",
                    "line_orders": [line.order for line in row_lines],
                    "bbox": self._merge_bboxes([line.bbox for line in row_lines if line.bbox]),
                    "rec_scores": [line.rec_score for line in row_lines if line.rec_score is not None],
                }
            )

        return rows

    def _join_row_text(self, row_lines: list[OCRLine], *, page_left: int) -> str:
        if not row_lines:
            return ""

        char_width = self._estimate_row_char_width(row_lines)
        parts: list[str] = []
        cursor_x = page_left

        for line in row_lines:
            text = line.normalized_text
            if not text:
                continue
            if not line.bbox:
                if parts:
                    parts.append(" ")
                parts.append(text)
                continue

            gap = max(0, line.bbox[0] - cursor_x)
            spaces = int(round(gap / char_width)) if char_width else 1
            if parts:
                spaces = max(1, spaces)
            else:
                spaces = max(0, spaces)
            if spaces:
                parts.append(" " * min(spaces, 80))
            parts.append(text)
            cursor_x = max(cursor_x, line.bbox[2])

        return "".join(parts).rstrip()

    def _estimate_page_skew_angle(self, lines: list[OCRLine]) -> float:
        angles: list[float] = []
        for line in lines:
            if not line.polygon or len(line.polygon) < 2:
                continue
            p0, p1 = line.polygon[0], line.polygon[1]
            if p0 == p1:
                continue
            angles.append(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))
        if not angles:
            return 0.0
        return float(median(angles))

    def _line_anchor_point(self, line: OCRLine) -> tuple[float, float]:
        if line.polygon and len(line.polygon) >= 2:
            left = line.polygon[0]
            right = line.polygon[1]
            return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        if line.bbox:
            return ((line.bbox[0] + line.bbox[2]) / 2.0, float(line.bbox[1]))
        return (0.0, 0.0)

    def _project_x(self, x: float, y: float, angle_deg: float) -> float:
        theta = math.radians(angle_deg)
        return x * math.cos(theta) + y * math.sin(theta)

    def _project_y(self, x: float, y: float, angle_deg: float) -> float:
        theta = math.radians(angle_deg)
        return y * math.cos(theta) - x * math.sin(theta)

    def _estimate_row_char_width(self, row_lines: list[OCRLine]) -> float:
        widths: list[float] = []
        for line in row_lines:
            if not line.bbox or not line.normalized_text:
                continue
            text_width = max(1, line.bbox[2] - line.bbox[0])
            widths.append(text_width / max(1, len(line.normalized_text)))
        if not widths:
            median_height = self._median_line_height(row_lines)
            return max(4.0, median_height * 0.5)
        return min(24.0, max(4.0, float(median(widths))))

    def _build_page_warnings(
        self,
        *,
        lines: list[OCRLine],
        page_index: int,
    ) -> list[dict[str, Any]]:
        warnings_out: list[dict[str, Any]] = []
        accepted_lines = [line for line in lines if line.accepted and line.normalized_text]
        if not accepted_lines:
            warnings_out.append(
                self._warning(
                    code="empty_page",
                    severity="high",
                    message="OCR produced no accepted text lines for this page.",
                    page_index=page_index,
                )
            )
            return warnings_out

        low_conf_count = len([line for line in accepted_lines if self._is_low_confidence(line)])
        if low_conf_count >= max(2, len(accepted_lines) // 3):
            warnings_out.append(
                self._warning(
                    code="many_low_confidence_lines",
                    severity="medium",
                    message="A large fraction of accepted OCR lines have low confidence.",
                    page_index=page_index,
                )
            )

        return warnings_out

    def _blocks_to_full_text(self, blocks: list[OCRBlock]) -> str:
        parts = [block.text.strip() for block in blocks if block.text.strip()]
        return "\n".join(parts)

    def _page_to_markdown(self, blocks: list[OCRBlock], *, fallback_text: str = "") -> str:
        layout_text = self._blocks_to_full_text(blocks) or fallback_text
        return self._text_fence(layout_text.rstrip())

    def _text_fence(self, text: str) -> str:
        max_backtick_run = max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0)
        fence = "`" * max(3, max_backtick_run + 1)
        return f"{fence}text\n{text}\n{fence}"

    def _line_sort_key(self, line: OCRLine) -> tuple[float, float, int]:
        if not line.polygon:
            return (float("inf"), float("inf"), line.order)
        xs = [point[0] for point in line.polygon]
        ys = [point[1] for point in line.polygon]
        return (float(min(ys)), float(min(xs)), line.order)

    def _coerce_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _list_item(self, values: list[Any], index: int) -> Any | None:
        if index >= len(values):
            return None
        return values[index]

    def _maybe_float(self, values: list[Any], index: int) -> float | None:
        if index >= len(values):
            return None
        try:
            return float(values[index])
        except (TypeError, ValueError):
            return None

    def _polygon_to_ints(self, polygon: Any) -> list[list[int]] | None:
        if polygon is None:
            return None
        try:
            points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
            out: list[list[int]] = []
            for pt in points:
                if len(pt) != 2:
                    continue
                out.append([int(pt[0]), int(pt[1])])
            return out or None
        except Exception:
            return None

    def _coerce_page_index(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _normalize_text(self, text: str, *, preserve_newlines: bool = False) -> str:
        if not text:
            return ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if preserve_newlines:
            normalized = "\n".join(self._normalize_text(part, preserve_newlines=False) for part in normalized.split("\n"))
            normalized = re.sub(r"\n{3,}", "\n\n", normalized)
            return normalized.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"\s+([,.;:!?%])", r"\1", normalized)
        normalized = re.sub(r"([(\[{])\s+", r"\1", normalized)
        normalized = re.sub(r"\s+([)\]}])", r"\1", normalized)
        return normalized.strip()

    def _polygon_to_bbox(self, polygon: list[list[int]] | None) -> list[int] | None:
        if not polygon:
            return None
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _merge_bboxes(self, bboxes: list[list[int] | None]) -> list[int] | None:
        valid = [bbox for bbox in bboxes if bbox]
        if not valid:
            return None
        return [
            min(bbox[0] for bbox in valid),
            min(bbox[1] for bbox in valid),
            max(bbox[2] for bbox in valid),
            max(bbox[3] for bbox in valid),
        ]

    def _bbox_center_y(self, bbox: list[int] | None) -> float:
        if not bbox:
            return 0.0
        return (bbox[1] + bbox[3]) / 2.0

    def _bbox_iou(self, left: list[int] | None, right: list[int] | None) -> float:
        if not left or not right:
            return 0.0
        inter_left = max(left[0], right[0])
        inter_top = max(left[1], right[1])
        inter_right = min(left[2], right[2])
        inter_bottom = min(left[3], right[3])
        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0
        intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
        left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
        right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
        union = left_area + right_area - intersection
        return intersection / float(union)

    def _median_line_height(self, lines: list[OCRLine]) -> float:
        heights = [max(1, line.bbox[3] - line.bbox[1]) for line in lines if line.bbox]
        if not heights:
            return 16.0
        return float(median(heights))

    def _mean_score(self, values: Any) -> float | None:
        floats = [float(value) for value in values if value is not None]
        if not floats:
            return None
        return round(sum(floats) / float(len(floats)), 4)

    def _mean_value(self, values: Any) -> float:
        floats = [float(value) for value in values]
        if not floats:
            return 0.0
        return sum(floats) / float(len(floats))

    def _is_low_confidence(self, line: OCRLine) -> bool:
        return line.rec_score is not None and line.rec_score < 0.8

    def _is_short_marker_line(self, line: OCRLine) -> bool:
        return line.normalized_text.lower() in {"o", "0", "x", "v", "□", "○", "◯", "•", "●"}

    def _is_duplicate_line(self, line: OCRLine, candidate: OCRLine) -> bool:
        if not line.normalized_text or line.normalized_text != candidate.normalized_text:
            return False
        if len(line.normalized_text) <= 2:
            return False
        return self._bbox_iou(line.bbox, candidate.bbox) >= 0.85

    def _warning(
        self,
        *,
        code: str,
        severity: str,
        message: str,
        page_index: int | None,
        line_orders: list[int] | None = None,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "severity": severity,
            "message": message,
            "page_index": page_index,
            "line_orders": list(line_orders or []),
        }
