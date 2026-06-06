from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from runtime_env import load_default_model_env


load_default_model_env()

def default_model_dir(name: str) -> Path:
    return Path.home() / "models" / "ocr" / name


def env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def env_str(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            return raw
    return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return values or default


def normalize_profile(profile: str | None) -> str:
    normalized = (profile or "").strip().lower()
    if normalized in {"full", "quality"}:
        return "full"
    if normalized in {"fast", "speed", ""}:
        return "fast"
    return "fast"


@dataclass(slots=True)
class OCRRuntimeConfig:
    profile: str
    device: str | None
    engine: str
    enable_hpi: bool
    use_tensorrt: bool
    run_mode: str
    trt_profile: str
    trt_modules: tuple[str, ...]
    trt_workspace_mb: int
    trt_det_max_side: int
    trt_det_opt_side: int
    trt_rec_max_width: int
    doc_orientation_model_name: str
    doc_orientation_model_dir: Path
    doc_unwarping_model_name: str
    doc_unwarping_model_dir: Path
    textline_orientation_model_name: str
    textline_orientation_model_dir: Path
    det_model_name: str
    det_model_dir: Path
    rec_model_name: str
    rec_model_dir: Path
    layout_detection_model_name: str
    layout_detection_model_dir: Path
    region_detection_model_name: str
    region_detection_model_dir: Path
    formula_recognition_model_name: str
    formula_recognition_model_dir: Path
    use_doc_orientation_classify: bool
    use_doc_unwarping: bool
    use_textline_orientation: bool
    use_document_structure: bool
    use_layout_detection: bool
    use_region_detection: bool
    use_formula_recognition: bool
    text_det_limit_side_len: int
    text_det_limit_type: str
    text_recognition_batch_size: int
    textline_orientation_batch_size: int
    formula_recognition_batch_size: int
    layout_threshold: float | None
    structured_markdown_mode: str

    @classmethod
    def from_env(
        cls,
        *,
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
    ) -> "OCRRuntimeConfig":
        normalized_profile = normalize_profile(profile or os.environ.get("OCR_PROFILE"))
        default_optional_modules = normalized_profile == "full"
        requested_tensorrt = use_tensorrt if use_tensorrt is not None else env_bool("OCR_USE_TENSORRT", False)
        run_mode = env_str(("OCR_RUN_MODE",), "trt_fp16" if requested_tensorrt else "paddle")
        use_document_structure = env_bool("OCR_USE_DOCUMENT_STRUCTURE", True)

        layout_threshold_raw = os.environ.get("OCR_LAYOUT_THRESHOLD")
        layout_threshold: float | None = None
        if layout_threshold_raw:
            try:
                layout_threshold = float(layout_threshold_raw)
            except ValueError:
                layout_threshold = None

        return cls(
            profile=normalized_profile,
            device=device or os.environ.get("OCR_DEVICE"),
            engine=engine or os.environ.get("OCR_ENGINE") or "paddle_static",
            enable_hpi=enable_hpi if enable_hpi is not None else env_bool("OCR_ENABLE_HPI", False),
            use_tensorrt=requested_tensorrt,
            run_mode=run_mode,
            trt_profile=(trt_profile or os.environ.get("OCR_TRT_PROFILE") or "jetson").strip().lower(),
            trt_modules=trt_modules or env_list("OCR_TRT_MODULES", ("det", "rec")),
            trt_workspace_mb=env_int(("OCR_TRT_WORKSPACE_MB",), 384),
            trt_det_max_side=env_int(("OCR_TRT_DET_MAX_SIDE",), 960),
            trt_det_opt_side=env_int(("OCR_TRT_DET_OPT_SIDE",), 256),
            trt_rec_max_width=env_int(("OCR_TRT_REC_MAX_WIDTH",), 1600),
            doc_orientation_model_name="PP-LCNet_x1_0_doc_ori",
            doc_orientation_model_dir=Path(
                doc_orientation_model_dir
                or os.environ.get("OCR_DOC_ORI_MODEL_DIR")
                or default_model_dir("PP-LCNet_x1_0_doc_ori_infer")
            ),
            doc_unwarping_model_name="UVDoc",
            doc_unwarping_model_dir=Path(
                doc_unwarping_model_dir
                or os.environ.get("OCR_DOC_UNWARP_MODEL_DIR")
                or default_model_dir("UVDoc_infer")
            ),
            textline_orientation_model_name="PP-LCNet_x0_25_textline_ori",
            textline_orientation_model_dir=Path(
                textline_orientation_model_dir
                or os.environ.get("OCR_TEXTLINE_ORI_MODEL_DIR")
                or default_model_dir("PP-LCNet_x0_25_textline_ori_infer")
            ),
            det_model_name="PP-OCRv5_mobile_det",
            det_model_dir=Path(
                det_model_dir
                or os.environ.get("OCR_DET_MODEL_DIR")
                or default_model_dir("PP-OCRv5_mobile_det_infer")
            ),
            rec_model_name="PP-OCRv5_mobile_rec",
            rec_model_dir=Path(
                rec_model_dir
                or os.environ.get("OCR_REC_MODEL_DIR")
                or default_model_dir("PP-OCRv5_mobile_rec_infer")
            ),
            layout_detection_model_name=os.environ.get("OCR_LAYOUT_MODEL_NAME", "PP-DocLayout_plus-L"),
            layout_detection_model_dir=Path(
                layout_detection_model_dir
                or os.environ.get("OCR_LAYOUT_MODEL_DIR")
                or default_model_dir("PP-DocLayout_plus-L_infer")
            ),
            region_detection_model_name=os.environ.get("OCR_REGION_MODEL_NAME", "PP-DocBlockLayout"),
            region_detection_model_dir=Path(
                region_detection_model_dir
                or os.environ.get("OCR_REGION_MODEL_DIR")
                or default_model_dir("PP-DocBlockLayout_infer")
            ),
            formula_recognition_model_name=os.environ.get("OCR_FORMULA_MODEL_NAME", "PP-FormulaNet_plus-S"),
            formula_recognition_model_dir=Path(
                formula_recognition_model_dir
                or os.environ.get("OCR_FORMULA_MODEL_DIR")
                or default_model_dir("PP-FormulaNet_plus-S_infer")
            ),
            use_doc_orientation_classify=env_bool("OCR_USE_DOC_ORIENTATION_CLASSIFY", default_optional_modules),
            use_doc_unwarping=env_bool("OCR_USE_DOC_UNWARPING", default_optional_modules),
            use_textline_orientation=env_bool("OCR_USE_TEXTLINE_ORIENTATION", default_optional_modules),
            use_document_structure=use_document_structure,
            use_layout_detection=env_bool("OCR_USE_LAYOUT_DETECTION", use_document_structure),
            use_region_detection=env_bool("OCR_USE_REGION_DETECTION", use_document_structure),
            use_formula_recognition=env_bool("OCR_USE_FORMULA_RECOGNITION", use_document_structure),
            text_det_limit_side_len=env_int(("OCR_TEXT_DET_LIMIT_SIDE_LEN", "OCR_DET_LIMIT_SIDE_LEN"), 64),
            text_det_limit_type=env_str(("OCR_TEXT_DET_LIMIT_TYPE", "OCR_DET_LIMIT_TYPE"), "min"),
            text_recognition_batch_size=text_recognition_batch_size or env_int(
                ("OCR_TEXT_RECOGNITION_BATCH_SIZE", "OCR_REC_BATCH_SIZE"),
                4,
            ),
            textline_orientation_batch_size=textline_orientation_batch_size or env_int(
                ("OCR_TEXTLINE_ORI_BATCH_SIZE",),
                6,
            ),
            formula_recognition_batch_size=formula_recognition_batch_size or env_int(
                ("OCR_FORMULA_RECOGNITION_BATCH_SIZE",),
                1,
            ),
            layout_threshold=layout_threshold,
            structured_markdown_mode=env_str(("OCR_STRUCTURED_MARKDOWN_MODE",), "prefer").strip().lower(),
        )
