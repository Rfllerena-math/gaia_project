"""Tavily web-search tool for Agent 4."""

from __future__ import annotations

import requests
from langchain_core.tools import tool

from ._tavily_client import TavilyConfigurationError, search_tavily


@tool
def tavily_search(query: str, max_results: int = 8) -> str:
    """Find relevant webpages with Tavily advanced search.

    This is a discovery tool: it returns titles, URLs, relevance scores, and short relevant
    chunks. For an evidence-rich answer, pass promising URLs to tavily_extract or use
    research_web to perform search and extraction together.
    """
    try:
        results = search_tavily(query, max_results=max_results)
    except (TavilyConfigurationError, ValueError, requests.RequestException) as error:
        return f"Tavily Search failed: {error}"

    formatted = []
    for index, item in enumerate(results, start=1):
        score = item.get("score")
        score_text = f" | relevance={score:.3f}" if isinstance(score, (int, float)) else ""
        formatted.append(
            f"[{index}] {item.get('title', '')}{score_text}\n"
            f"URL: {item.get('url', '')}\n"
            f"Relevant content: {item.get('content', '')}"
        )
    return "\n\n".join(formatted) or "Tavily Search returned no results."
