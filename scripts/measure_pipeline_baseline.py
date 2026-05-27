from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from ocr_service.pipeline import OCRPipeline

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure OCRPipeline baseline timing on input files.")
    parser.add_argument("--input", default="test_set", help="Input file or directory.")
    parser.add_argument(
        "--output",
        default="data/ocr_benchmarks/pipeline_baseline_test_set.json",
        help="Output JSON report path.",
    )
    parser.add_argument("--profile", choices=("fast", "full"), default="fast")
    parser.add_argument("--engine", default="paddle_static")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = collect_inputs(input_path)

    pipeline = OCRPipeline(
        profile=args.profile,
        engine=args.engine,
        device=args.device,
        use_tensorrt=False,
        enable_hpi=False,
        precision="fp32",
    )

    if inputs and args.warmup_runs > 0:
        for _ in range(args.warmup_runs):
            pipeline.predict_document(inputs[0])

    files: list[dict[str, object]] = []
    total_pages = 0
    total_pipeline_ms = 0.0

    wall_total_start = perf_counter()

    for file_path in inputs:
        wall_start = perf_counter()
        results = pipeline.predict_document(file_path)
        wall_ms = (perf_counter() - wall_start) * 1000.0

        page_count = len(results)
        per_page_total_ms = [float((result.timings_ms or {}).get("total") or 0.0) for result in results]
        pipeline_ms = sum(per_page_total_ms)

        total_pages += page_count
        total_pipeline_ms += pipeline_ms

        files.append(
            {
                "path": str(file_path),
                "pages": page_count,
                "pipeline_ms": round(pipeline_ms, 2),
                "wall_ms": round(wall_ms, 2),
                "page_totals_ms": [round(value, 2) for value in per_page_total_ms],
            }
        )

        print(f"processed {file_path.name}: pages={page_count} pipeline_ms={pipeline_ms:.2f} wall_ms={wall_ms:.2f}")

    wall_total_ms = (perf_counter() - wall_total_start) * 1000.0

    report = {
        "settings": {
            "profile": args.profile,
            "engine": args.engine,
            "device": args.device,
            "use_tensorrt": False,
            "enable_hpi": False,
            "precision": "fp32",
            "warmup_runs": args.warmup_runs,
        },
        "summary": {
            "files": len(inputs),
            "pages": total_pages,
            "pipeline_total_ms": round(total_pipeline_ms, 2),
            "pipeline_avg_ms_per_page": round(total_pipeline_ms / max(1, total_pages), 2),
            "wall_total_ms": round(wall_total_ms, 2),
            "wall_avg_ms_per_page": round(wall_total_ms / max(1, total_pages), 2),
        },
        "files_detail": files,
    }

    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print("summary:")
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    print(f"saved report: {output_path}")


if __name__ == "__main__":
    main()
