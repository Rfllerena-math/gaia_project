"""Exa neural-search tool used by the research and recovery agents."""

from __future__ import annotations

import os

import requests
from langchain_core.tools import tool

EXA_URL = "https://api.exa.ai/search"


@tool
def exa_search(query: str) -> str:
    """Search the web with Exa and return up to five titles, URLs, and text snippets."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return "Exa search failed: EXA_API_KEY is not configured."
    try:
        response = requests.post(
            EXA_URL,
            json={"query": query, "numResults": 5, "contents": {"text": True}},
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return f"Exa search failed: {error}"

    results = response.json().get("results", [])
    formatted_results = [
        f"{item.get('title', '')}\n{item.get('url', '')}\n{(item.get('text') or '')[:500]}"
        for item in results
    ]
    return "\n\n".join(formatted_results) or "No results found."
