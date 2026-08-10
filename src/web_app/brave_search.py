"""Small backend-only Brave LLM-context client.

The web application intentionally keeps this integration private: the
browser only sends ``search_web`` and never receives the API credential.  The
request shape mirrors the F1 fact-checker client, including Brave's
``/v1/llm/context`` query parameters and ``X-Subscription-Token`` header.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_BRAVE_LLM_CONTEXT_ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"
DEFAULT_BRAVE_SEARCH_COUNT = 5
DEFAULT_BRAVE_SEARCH_TIMEOUT = 10.0
DEFAULT_BRAVE_CONTEXT_MAX_URLS = 3
DEFAULT_BRAVE_CONTEXT_MAX_SNIPPETS = 8
DEFAULT_BRAVE_CONTEXT_MAX_TOKENS = 3000
MAX_QUERY_CHARS = 400
MAX_QUERY_WORDS = 50


class BraveSearchError(RuntimeError):
    """Raised when Brave is unavailable or returns unusable context."""


@dataclass(frozen=True, slots=True)
class BraveSearchConfig:
    context_endpoint: str = DEFAULT_BRAVE_LLM_CONTEXT_ENDPOINT
    count: int = DEFAULT_BRAVE_SEARCH_COUNT
    timeout_seconds: float = DEFAULT_BRAVE_SEARCH_TIMEOUT
    max_urls: int = DEFAULT_BRAVE_CONTEXT_MAX_URLS
    max_snippets: int = DEFAULT_BRAVE_CONTEXT_MAX_SNIPPETS
    max_tokens: int = DEFAULT_BRAVE_CONTEXT_MAX_TOKENS
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "BraveSearchConfig":
        return cls(
            context_endpoint=os.environ.get(
                "BRAVE_LLM_CONTEXT_ENDPOINT", DEFAULT_BRAVE_LLM_CONTEXT_ENDPOINT
            ).strip()
            or DEFAULT_BRAVE_LLM_CONTEXT_ENDPOINT,
            count=_positive_int("BRAVE_CONTEXT_COUNT", _positive_int("BRAVE_SEARCH_COUNT", DEFAULT_BRAVE_SEARCH_COUNT)),
            timeout_seconds=_positive_float("BRAVE_SEARCH_TIMEOUT", DEFAULT_BRAVE_SEARCH_TIMEOUT),
            max_urls=_positive_int("BRAVE_CONTEXT_MAX_URLS", DEFAULT_BRAVE_CONTEXT_MAX_URLS),
            max_snippets=_positive_int("BRAVE_CONTEXT_MAX_SNIPPETS", DEFAULT_BRAVE_CONTEXT_MAX_SNIPPETS),
            max_tokens=_positive_int("BRAVE_CONTEXT_MAX_TOKENS", DEFAULT_BRAVE_CONTEXT_MAX_TOKENS),
            api_key=_api_key_from_environment_or_local_env(),
        )

    def require_api_key(self) -> str:
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        raise BraveSearchError("Web search is not configured on the backend.")


def _positive_int(name: str, fallback: int) -> int:
    try:
        value = int(os.environ.get(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback
    return max(value, 1)


def _positive_float(name: str, fallback: float) -> float:
    try:
        value = float(os.environ.get(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback
    return max(value, 0.1)


def _api_key_from_environment_or_local_env() -> str | None:
    """Read only ``BRAVE_SEARCH_API_KEY`` from the local F1 .env fallback."""
    configured = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if configured:
        return configured

    source_root = Path(__file__).resolve().parents[2]
    candidates = (
        source_root.parent / "F1_fact_checker" / ".env",
        Path.cwd().parent / "F1_fact_checker" / ".env",
    )
    seen: set[Path] = set()
    for env_path in candidates:
        env_path = env_path.resolve()
        if env_path in seen or not env_path.is_file():
            continue
        seen.add(env_path)
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                match = re.match(
                    r"^\s*(?:export\s+)?BRAVE_SEARCH_API_KEY\s*=\s*(.*?)\s*$",
                    raw_line,
                )
                if not match:
                    continue
                value = match.group(1).strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                # A local .env may contain an inline comment.  API keys do
                # not contain whitespace, so this safely strips only comments
                # separated by whitespace without logging the value.
                value = value.split(" #", 1)[0].strip()
                if value:
                    return value
        except OSError:
            continue
    return None


class BraveSearchClient:
    def __init__(self, config: BraveSearchConfig | None = None) -> None:
        self.config = config or BraveSearchConfig.from_env()

    def llm_context(
        self,
        query: str,
        *,
        count: int | None = None,
        max_urls: int | None = None,
        max_snippets: int | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, str | None]]:
        normalized_query = bound_query(query)
        if not normalized_query:
            raise BraveSearchError("Search query must not be empty.")
        self.config.require_api_key()
        params = {
            "q": normalized_query,
            "count": _bounded_positive(count, self.config.count),
            "maximum_number_of_urls": _bounded_positive(max_urls, self.config.max_urls),
            "maximum_number_of_snippets": _bounded_positive(max_snippets, self.config.max_snippets),
            "maximum_number_of_tokens": _bounded_positive(max_tokens, self.config.max_tokens),
            "enable_source_metadata": "true",
            "context_threshold_mode": "balanced",
        }
        encoded = urllib.parse.urlencode(params)
        url = f"{self.config.context_endpoint}?{encoded}"
        req = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "X-Subscription-Token": self.config.require_api_key(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BraveSearchError(f"Brave Search API request failed with HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BraveSearchError("Brave Search API is unavailable.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise BraveSearchError("Brave Search API returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise BraveSearchError("Brave Search API returned an unexpected payload.")
        return normalize_context_results(payload, max_urls=params["maximum_number_of_urls"])


def _bounded_positive(value: int | None, fallback: int) -> int:
    if value is None:
        return max(int(fallback), 1)
    return max(min(int(value), 100), 1)


def bound_query(query: str) -> str:
    """Apply Brave's 400-character/50-word LLM-context query limits."""
    words = " ".join(str(query or "").strip().split())
    if not words:
        return ""
    words = words[:MAX_QUERY_CHARS].rstrip()
    return " ".join(words.split()[:MAX_QUERY_WORDS])


def search_llm_context(
    query: str,
    *,
    count: int | None = None,
    max_urls: int | None = None,
    max_snippets: int | None = None,
    max_tokens: int | None = None,
    client: BraveSearchClient | None = None,
) -> list[dict[str, str | None]]:
    return (client or BraveSearchClient()).llm_context(
        query,
        count=count,
        max_urls=max_urls,
        max_snippets=max_snippets,
        max_tokens=max_tokens,
    )


def normalize_context_results(
    payload: dict[str, Any], *, max_urls: int | None = None
) -> list[dict[str, str | None]]:
    """Normalize Brave's generic grounding items to the F1 source shape."""
    grounding = payload.get("grounding")
    generic = grounding.get("generic") if isinstance(grounding, dict) else None
    sources = payload.get("sources")
    results: list[dict[str, str | None]] = []
    if isinstance(generic, list):
        for item in generic:
            if not isinstance(item, dict):
                continue
            url = _clean(item.get("url"))
            title = _clean(item.get("title"))
            snippets = item.get("snippets")
            if not url or not title or not _valid_http_url(url) or not isinstance(snippets, list):
                continue
            cleaned = [_clean(snippet) for snippet in snippets]
            cleaned = [snippet for snippet in cleaned if snippet]
            if not cleaned:
                continue
            metadata = sources.get(url) if isinstance(sources, dict) else None
            metadata = metadata if isinstance(metadata, dict) else {}
            source = _clean(metadata.get("site_name")) or _domain(url)
            results.append(
                {
                    "title": title,
                    "url": url,
                    "description": "\n\n".join(cleaned),
                    "snippet": "\n\n".join(cleaned),
                    "source": source,
                    "published_at": _clean(metadata.get("page_last_modified")),
                }
            )
    else:
        # Keep the adapter tolerant of Brave search-shaped payloads used by
        # local mocks and older API responses.
        raw = payload.get("results")
        if not isinstance(raw, list):
            container = payload.get("web")
            raw = container.get("results") if isinstance(container, dict) else []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                url = _clean(item.get("url"))
                title = _clean(item.get("title"))
                snippet = _clean(item.get("description") or item.get("snippet"))
                if not url or not title or not snippet or not _valid_http_url(url):
                    continue
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "description": snippet,
                        "snippet": snippet,
                        "source": _domain(url),
                        "published_at": _clean(item.get("published_at") or item.get("date")),
                    }
                )
    if max_urls is not None:
        results = results[: max(int(max_urls), 1)]
    return results


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _domain(value: str) -> str | None:
    hostname = urlparse(value).hostname
    return hostname.removeprefix("www.") if hostname else None
