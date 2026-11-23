"""Search-select-extract orchestration for evidence-rich Agent 4 research."""

from __future__ import annotations

import requests
from langchain_core.tools import tool

from ._tavily_client import TavilyConfigurationError, extract_tavily, search_tavily


@tool
def research_web(question: str, max_sources: int = 5) -> str:
    """Research a question using Tavily search followed by semantic page extraction.

    Use this as the default tool for external factual questions. It searches broadly, ranks and
    deduplicates sources, then extracts up to five question-relevant chunks from each selected
    page. The output contains source URLs and evidence rather than a generated Tavily answer.
    """
    if not 1 <= max_sources <= 8:
        return "Web research failed: max_sources must be between 1 and 8."

    try:
        search_results = search_tavily(
            question,
            max_results=max(max_sources * 2, 8),
            search_depth="advanced",
        )
        ranked = sorted(
            search_results,
            key=lambda item: float(item.get("score") or 0),
            reverse=True,
        )
        selected = ranked[:max_sources]
        urls = [str(item["url"]) for item in selected]
        if not urls:
            return "Web research found no usable sources."

        extracted, failures = extract_tavily(
            urls,
            query=question,
            chunks_per_source=5,
            extract_depth="advanced",
        )
    except (TavilyConfigurationError, ValueError, requests.RequestException) as error:
        return f"Web research failed: {error}"

    search_by_url = {str(item["url"]): item for item in selected}
    evidence = []
    for index, item in enumerate(extracted, start=1):
        url = str(item.get("url", ""))
        search_item = search_by_url.get(url, {})
        content = (item.get("raw_content") or "").strip()
        evidence.append(
            f"SOURCE {index}\n"
            f"Title: {search_item.get('title', '')}\n"
            f"URL: {url}\n"
            f"Search relevance: {search_item.get('score', 'unknown')}\n"
            f"Question-relevant evidence:\n{content or search_item.get('content', '')}"
        )

    if failures:
        failed_urls = ", ".join(str(item.get("url", "")) for item in failures)
        evidence.append(f"Extraction failed for: {failed_urls}")

    return (
        f"Tavily researched {len(search_results)} search results and extracted "
        f"{len(extracted)} selected sources.\n\n" + "\n\n---\n\n".join(evidence)
    )
