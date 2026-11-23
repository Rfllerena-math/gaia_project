"""Web-search tool."""

from __future__ import annotations

import os

import requests
from ddgs import DDGS
from langchain_core.tools import tool

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 10


def _format_results(results: list[dict[str, str]]) -> str:
    """Return provider results in the common title, URL, snippet format."""
    return "\n\n".join(
        f"{result.get('title', '')}\n{result.get('url', '')}\n{result.get('content', '')}"
        for result in results
    ) or "No results found."


def _search_duckduckgo(query: str) -> str:
    """Search DuckDuckGo and normalize its result schema."""
    results = DDGS().text(query, max_results=MAX_RESULTS)
    return _format_results(
        [
            {
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "content": result.get("body", ""),
            }
            for result in results
        ]
    )


def _search_brave(query: str, api_key: str) -> str:
    """Search Brave and normalize its response."""
    response = requests.get(
        BRAVE_URL,
        params={"q": query, "count": MAX_RESULTS},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("web", {}).get("results", [])
    return _format_results(
        [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("description", ""),
            }
            for result in results
        ]
    )


@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo, then Brave if needed, returning up to ten results."""
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    errors: list[str] = []
    try:
        return _search_duckduckgo(query)
    except Exception as error:
        errors.append(f"DuckDuckGo: {error}")
    if brave_key:
        try:
            return _search_brave(query, brave_key)
        except requests.RequestException as error:
            errors.append(f"Brave: {error}")
    else:
        errors.append("Brave: BRAVE_SEARCH_API_KEY is not configured")
    return f"Web search failed: {'; '.join(errors)}"