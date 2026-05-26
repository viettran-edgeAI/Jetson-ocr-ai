from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import OCRRuntimeConfig


def _as_list(output: Any) -> list[Any]:
    if output is None:
        return []
    if isinstance(output, list):
        return output
    if isinstance(output, tuple):
        return list(output)
    if hasattr(output, "__iter__") and not isinstance(output, (dict, str, bytes)):
        try:
            return list(output)
        except TypeError:
            return [output]
    return [output]


def _to_dict(item: Any) -> dict[str, Any]:
    if item is None:
        return {}
    if isinstance(item, dict):
        return item
    if hasattr(item, "res"):
        res_value = getattr(item, "res")
        if isinstance(res_value, dict):
            return res_value
    if hasattr(item, "to_dict"):
        try:
            out = item.to_dict()
            if isinstance(out, dict):
                return out.get("res", out)
        except Exception:
            return {}
    return {}


def _maybe_predict(model: Any, data: Any, batch_size: int | None = None) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    try:
        out = model.predict(data, **kwargs)
    except TypeError:
        out = model.predict(input=data, **kwargs)
    return [_to_dict(x) for x in _as_list(out)]


@dataclass
class PaddleModules:
    doc_orientation: Any | None
    doc_unwarping: Any | None
    layout_detection: Any | None
    region_detection: Any | None
    formula_recognition: Any | None
    text_detection: Any | None
    textline_orientation: Any | None
    text_recognition: Any | None


class PaddleRuntime:
    def __init__(self, config: OCRRuntimeConfig) -> None:
        self.config = config
        self.modules = PaddleModules(
            doc_orientation=None,
            doc_unwarping=None,
            layout_detection=None,
            region_detection=None,
            formula_recognition=None,
            text_detection=None,
            textline_orientation=None,
            text_recognition=None,
        )
        self._classes: dict[str, Any] | None = None

    def _module_kwargs(self, model_dir: Path, model_name: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model_dir": str(model_dir), "model_name": model_name}
        if self.config.device:
            kwargs["device"] = self.config.device
        if self.config.engine:
            kwargs["engine"] = self.config.engine
        return kwargs

    @staticmethod
    def _usable_model_dir(model_dir: Path) -> bool:
        if not model_dir.exists():
            return False
        return any((model_dir / name).exists() for name in ("inference.yml", "inference.json", "config.json"))

    def _load_classes(self) -> dict[str, Any]:
        if self._classes is not None:
            return self._classes
        try:
            from paddleocr import (
                DocImgOrientationClassification,
                FormulaRecognition,
                LayoutDetection,
                TextDetection,
                TextImageUnwarping,
                TextLineOrientationClassification,
                TextRecognition,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not available in this environment. Install the OCR service dependencies first."
            ) from exc

        self._classes = {
            "doc_orientation": DocImgOrientationClassification,
            "doc_unwarping": TextImageUnwarping,
            "layout_detection": LayoutDetection,
            "region_detection": LayoutDetection,
            "formula_recognition": FormulaRecognition,
            "text_detection": TextDetection,
            "textline_orientation": TextLineOrientationClassification,
            "text_recognition": TextRecognition,
        }
        return self._classes

    @staticmethod
    def _model_name(attr: str) -> str:
        return {
            "doc_orientation": "PP-LCNet_x1_0_doc_ori",
            "doc_unwarping": "UVDoc",
            "layout_detection": "PP-DocLayout_plus-L",
            "region_detection": "PP-DocBlockLayout",
            "formula_recognition": "PP-FormulaNet_plus-S",
            "text_detection": "PP-OCRv5_mobile_det",
            "textline_orientation": "PP-LCNet_x0_25_textline_ori",
            "text_recognition": "PP-OCRv5_mobile_rec",
        }[attr]

    def _get_module(self, attr: str, model_dir: Path) -> Any | None:
        current = getattr(self.modules, attr)
        if current is not None:
            return current
        if not self._usable_model_dir(model_dir):
            return None
        cls = self._load_classes()[attr]
        module = cls(**self._module_kwargs(model_dir, self._model_name(attr)))
        setattr(self.modules, attr, module)
        return module

    def predict_doc_orientation(self, image_np: np.ndarray) -> dict[str, Any]:
        module = self._get_module("doc_orientation", self.config.doc_orientation_model_dir)
        if module is None:
            return {}
        items = _maybe_predict(module, image_np, batch_size=1)
        return items[0] if items else {}

    def predict_doc_unwarp(self, image_np: np.ndarray) -> dict[str, Any]:
        module = self._get_module("doc_unwarping", self.config.doc_unwarping_model_dir)
        if module is None:
            return {}
        items = _maybe_predict(module, image_np, batch_size=1)
        return items[0] if items else {}

    def predict_layout(self, image_np: np.ndarray) -> dict[str, Any]:
        module = self._get_module("layout_detection", self.config.layout_detection_model_dir)
        if module is None:
            return {}
        items = _maybe_predict(module, image_np, batch_size=1)
        return items[0] if items else {}

    def predict_regions(self, image_np: np.ndarray) -> dict[str, Any]:
        module = self._get_module("region_detection", self.config.region_detection_model_dir)
        if module is None:
            return {}
        items = _maybe_predict(module, image_np, batch_size=1)
        return items[0] if items else {}

    def predict_formula(self, crops: list[np.ndarray], batch_size: int = 1) -> list[dict[str, Any]]:
        if not crops:
            return []
        module = self._get_module("formula_recognition", self.config.formula_recognition_model_dir)
        if module is None:
            return []
        return _maybe_predict(module, crops, batch_size=batch_size)

    def predict_text_detection(self, image_np: np.ndarray) -> dict[str, Any]:
        module = self._get_module("text_detection", self.config.det_model_dir)
        if module is None:
            return {}
        items = _maybe_predict(module, image_np, batch_size=1)
        return items[0] if items else {}

    def predict_textline_orientation(self, crops: list[np.ndarray]) -> list[dict[str, Any]]:
        module = self._get_module("textline_orientation", self.config.textline_orientation_model_dir)
        if not crops or module is None:
            return []
        return _maybe_predict(module, crops, batch_size=len(crops))

    def predict_text_recognition(self, crops: list[np.ndarray], batch_size: int) -> list[dict[str, Any]]:
        module = self._get_module("text_recognition", self.config.rec_model_dir)
        if not crops or module is None:
            return []
        return _maybe_predict(module, crops, batch_size=batch_size)
