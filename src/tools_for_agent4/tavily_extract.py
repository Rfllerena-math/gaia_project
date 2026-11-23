"""Multi-page, query-aware Tavily extraction tool for Agent 4."""

from __future__ import annotations

import requests
from langchain_core.tools import tool

from ._tavily_client import TavilyConfigurationError, extract_tavily


@tool
def tavily_extract(
    urls: list[str],
    query: str,
    chunks_per_source: int = 5,
    extract_depth: str = "advanced",
) -> str:
    """Extract question-relevant chunks from one or more known webpages.

    Supply the original question as `query`. Tavily uses it to semantically rerank chunks from
    every URL. Up to 20 URLs and 1-5 chunks per source are supported. Use advanced depth for
    JavaScript pages, tables, or difficult layouts.
    """
    try:
        results, failures = extract_tavily(
            urls,
            query=query,
            chunks_per_source=chunks_per_source,
            extract_depth=extract_depth,
        )
    except (TavilyConfigurationError, ValueError, requests.RequestException) as error:
        return f"Tavily Extract failed: {error}"

    sections = []
    for item in results:
        content = (item.get("raw_content") or "").strip()
        sections.append(f"SOURCE: {item.get('url', '')}\n{content or '[No relevant content]'}")
    for item in failures:
        sections.append(
            f"FAILED SOURCE: {item.get('url', '')}\n{item.get('error', 'Extraction failed')}"
        )
    return "\n\n---\n\n".join(sections) or "Tavily Extract returned no content."
