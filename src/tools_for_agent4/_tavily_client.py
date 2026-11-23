"""Small internal client shared by the Agent 4 Tavily tools."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
REQUEST_TIMEOUT = 60


class TavilyConfigurationError(RuntimeError):
    """Raised when Tavily has not been configured."""


def _headers() -> dict[str, str]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise TavilyConfigurationError("TAVILY_API_KEY is not configured.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _validate_urls(urls: list[str]) -> list[str]:
    if not urls:
        raise ValueError("At least one URL is required.")
    if len(urls) > 20:
        raise ValueError("Tavily Extract accepts at most 20 URLs per request.")

    cleaned: list[str] = []
    for url in urls:
        normalized = url.strip()
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid HTTP(S) URL: {url}")
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def search_tavily(
    query: str,
    *,
    max_results: int = 8,
    search_depth: str = "advanced",
) -> list[dict[str, Any]]:
    """Return normalized Tavily search result dictionaries."""
    query = query.strip()
    if not query:
        raise ValueError("Search query must not be empty.")
    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20.")
    if search_depth not in {"basic", "advanced"}:
        raise ValueError("search_depth must be 'basic' or 'advanced'.")

    payload: dict[str, Any] = {
        "query": query,
        "topic": "general",
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    if search_depth == "advanced":
        payload["chunks_per_source"] = 3

    response = requests.post(
        TAVILY_SEARCH_URL,
        headers=_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return [item for item in results if isinstance(item, dict) and item.get("url")]


def extract_tavily(
    urls: list[str],
    *,
    query: str,
    chunks_per_source: int = 5,
    extract_depth: str = "advanced",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract and semantically rerank relevant chunks from multiple URLs."""
    urls = _validate_urls(urls)
    query = query.strip()
    if not query:
        raise ValueError("query is required to rerank extracted page chunks.")
    if not 1 <= chunks_per_source <= 5:
        raise ValueError("chunks_per_source must be between 1 and 5.")
    if extract_depth not in {"basic", "advanced"}:
        raise ValueError("extract_depth must be 'basic' or 'advanced'.")

    response = requests.post(
        TAVILY_EXTRACT_URL,
        headers=_headers(),
        json={
            "urls": urls,
            "query": query,
            "chunks_per_source": chunks_per_source,
            "extract_depth": extract_depth,
            "format": "markdown",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    results = [item for item in body.get("results", []) if isinstance(item, dict)]
    failures = [item for item in body.get("failed_results", []) if isinstance(item, dict)]
    return results, failures
