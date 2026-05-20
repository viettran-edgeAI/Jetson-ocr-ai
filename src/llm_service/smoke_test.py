from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


DEFAULT_MARKDOWN = Path("data/ocr_markdown_run_v2/documents/question_0.md")
DEFAULT_EXPECTED_ANSWER = "Product Owner"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the OCR LLM answer API.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/answer")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--request", default="answer question")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--expect-contains", default=DEFAULT_EXPECTED_ANSWER)
    args = parser.parse_args()

    ocr_markdown = args.markdown.read_text(encoding="utf-8")
    payload = {
        "ocr_markdown": ocr_markdown,
        "user_request": args.request,
        "max_tokens": args.max_tokens,
    }

    started = time.perf_counter()
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    answer = str(data.get("answer") or "")
    success = args.expect_contains.lower() in answer.lower()

    print(f"success: {success}")
    print(f"answer: {answer}")
    print(f"client_elapsed_ms: {elapsed_ms}")
    print(f"service_elapsed_ms: {data.get('elapsed_ms')}")
    print(f"model: {data.get('model')}")
    print(f"ocr_chars: {data.get('ocr_chars')}")
    print(f"ocr_truncated: {data.get('ocr_truncated')}")
    if data.get("total_tokens") is not None:
        print(f"total_tokens: {data['total_tokens']}")

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
