"""Tavily-backed answer verification used exclusively by Agent 4."""

from __future__ import annotations

import json
import re

import requests
from langchain_core.tools import tool

from ._tavily_client import TavilyConfigurationError, extract_tavily, search_tavily

NUMBER_REQUEST = re.compile(
    r"\b(?:how many|count|number|total|sum|percentage|percent|average|year|date)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d")
WORD = re.compile(r"[A-Za-z0-9]+")
IGNORED_WORDS = frozenset({"a", "an", "and", "is", "of", "the", "to"})


def _meaningful_terms(text: str) -> set[str]:
    return {
        word.lower()
        for word in WORD.findall(text)
        if len(word) > 1 and word.lower() not in IGNORED_WORDS
    }


@tool
def verify_answer(question: str, candidate_answer: str) -> str:
    """Check a candidate against independently retrieved Tavily evidence.

    Returns JSON containing `approved`, `issues`, `evidence_urls`, and `next_tool`.
    """
    issues: list[str] = []
    candidate = candidate_answer.strip()
    if not candidate:
        issues.append("The candidate answer is empty.")
    if NUMBER_REQUEST.search(question) and not NUMBER.search(candidate):
        issues.append("The question requests a numeric result but the candidate has no number.")

    evidence_urls: list[str] = []
    evidence_text = ""
    try:
        results = search_tavily(question, max_results=5, search_depth="advanced")
        urls = [str(item["url"]) for item in results[:3]]
        evidence_urls = urls
        if urls:
            extracted, _ = extract_tavily(
                urls,
                query=question,
                chunks_per_source=5,
                extract_depth="advanced",
            )
            evidence_text = " ".join(
                str(item.get("raw_content") or "") for item in extracted
            ).lower()
    except (TavilyConfigurationError, ValueError, requests.RequestException) as error:
        issues.append(f"Independent Tavily verification failed: {error}")

    if not evidence_urls:
        issues.append("The verification search returned no usable sources.")
    elif candidate:
        candidate_terms = _meaningful_terms(candidate)
        evidence_terms = set(WORD.findall(evidence_text))
        if candidate_terms and not candidate_terms.issubset(evidence_terms):
            issues.append("The extracted evidence does not fully support the candidate answer.")

    return json.dumps(
        {
            "approved": not issues,
            "issues": issues,
            "evidence_urls": evidence_urls,
            "next_tool": "final_answer" if not issues else "research_web",
        }
    )
