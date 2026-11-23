"""Tavily Extract tool for retrieving the full readable content of a known URL."""

from __future__ import annotations

import os

import requests
from langchain_core.tools import tool

TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


@tool
def tavily_extract(url: str) -> str:
    """Extract the readable content of a webpage using Tavily Extract, limited to 12,000 characters."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Tavily Extract failed: TAVILY_API_KEY is not configured."
    try:
        response = requests.post(
            TAVILY_EXTRACT_URL,
            json={"api_key": api_key, "urls": [url]},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return f"Tavily Extract failed: {error}"

    results = response.json().get("results", [])
    if not results:
        return "Tavily Extract returned no content."
    return results[0].get("raw_content", "")[:12000] or "The page contained no readable text."
