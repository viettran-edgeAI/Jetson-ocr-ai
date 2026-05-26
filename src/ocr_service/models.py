from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class OCRRegion:
    id: str
    order: int
    label: str
    bbox: list[int]
    page_index: int
    confidence: float | None = None
    source: str = "layout"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "order": self.order,
            "label": self.label,
            "bbox": list(self.bbox),
            "page_index": self.page_index,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(slots=True)
class OCRFormula:
    id: str
    order: int
    latex: str
    page_index: int
    bbox: list[int] | None = None
    confidence: float | None = None
    line_orders: list[int] = field(default_factory=list)
    source: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "order": self.order,
            "latex": self.latex,
            "page_index": self.page_index,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "line_orders": list(self.line_orders),
            "source": self.source,
        }


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
    regions: list[OCRRegion] = field(default_factory=list)
    formulas: list[OCRFormula] = field(default_factory=list)

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
            "regions": [region.to_dict() for region in self.regions],
            "formulas": [formula.to_dict() for formula in self.formulas],
            "warnings": [dict(warning) for warning in self.warnings],
            "timings_ms": dict(self.timings_ms),
            "meta": dict(self.meta),
        }
