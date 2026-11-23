"""Webpage extraction tool."""

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

DEFAULT_MAX_CHARS = 50000
MAX_CHARS = 100000
DEFAULT_ANALYSIS_CHARS = 60000
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
IGNORED_QUERY_WORDS = frozenset(
    {"a", "an", "and", "are", "by", "for", "from", "how", "in", "is", "of", "the", "to", "what", "which"}
)
REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36",
}


def _main_content(soup: BeautifulSoup):
    """Locate the article body while excluding site navigation and boilerplate."""
    return (
        soup.select_one("#mw-content-text .mw-parser-output")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.select_one('[role="main"]')
        or soup.body
        or soup
    )


def _readable_content(soup: BeautifulSoup) -> str:
    """Extract page content while excluding site navigation and other boilerplate."""
    content = _main_content(soup)
    for element in content(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside"]):
        element.decompose()
    return content.get_text("\n", strip=True)


def _page_sections(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Split the full main content into heading-delimited sections."""
    content = _main_content(soup)
    for element in content(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside"]):
        element.decompose()

    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    for element in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "table"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            sections.append((text, []))
        elif element.name == "table" and element.find_parent("table"):
            continue
        else:
            sections[-1][1].append(text)
    return [(title, "\n".join(parts)) for title, parts in sections if parts]


def _question_terms(question: str) -> set[str]:
    """Return meaningful question terms for deterministic section ranking."""
    return {
        word.lower()
        for word in WORD_PATTERN.findall(question)
        if len(word) > 1 and word.lower() not in IGNORED_QUERY_WORDS
    }


def _rank_sections(sections: list[tuple[str, str]], question: str) -> list[tuple[str, str]]:
    """Rank all sections by occurrences of meaningful question terms."""
    terms = _question_terms(question)
    if {"album", "albums"} & terms:
        terms.add("discography")
    if not terms:
        return sections

    def score(section: tuple[str, str]) -> int:
        title, text = section
        title_words = WORD_PATTERN.findall(title.lower())
        body_words = WORD_PATTERN.findall(text.lower())
        return 3 * sum(word in terms for word in title_words) + sum(word in terms for word in body_words)

    return sorted(sections, key=score, reverse=True)


@tool
def read_webpage(url: str, start_char: int = 0, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Download the main content of a webpage, optionally reading a later chunk.

    Use start_char to continue after a prior response when the required details are in later
    sections. The response includes the character range and total readable length.
    """
    if start_char < 0:
        return "start_char must be zero or greater."
    if not 1 <= max_chars <= MAX_CHARS:
        return f"max_chars must be between 1 and {MAX_CHARS}."
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as error:
        return (
            f"Unable to read {url}: {error}. "
            "Use web_search to find an accessible, authoritative alternative source."
        )
    soup = BeautifulSoup(response.text, "html.parser")
    text = _readable_content(soup)
    if not text:
        return "The page contained no readable text."
    if start_char >= len(text):
        return f"No text remains after character {start_char}; page length is {len(text)}."
    end_char = min(start_char + max_chars, len(text))
    return f"Content characters {start_char}-{end_char} of {len(text)}:\n{text[start_char:end_char]}"


@tool
def analyze_webpage(
    url: str, question: str, max_sections: int = 6, max_chars: int = DEFAULT_ANALYSIS_CHARS
) -> str:
    """Analyze every main-content section of a webpage against a question.

    This scans the complete article or page body, including sections after the introduction,
    then returns the most relevant heading-delimited sections. It analyzes one supplied URL,
    not every page linked from a website.
    """
    if max_sections < 1:
        return "max_sections must be at least 1."
    if not 1 <= max_chars <= MAX_CHARS:
        return f"max_chars must be between 1 and {MAX_CHARS}."
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as error:
        return f"Unable to analyze {url}: {error}"

    sections = _page_sections(BeautifulSoup(response.text, "html.parser"))
    if not sections:
        return "The page contained no readable main-content sections."
    selected_sections = _rank_sections(sections, question)[:max_sections]
    section_char_limit = max(1, max_chars // len(selected_sections))
    analysis = "\n\n".join(
        f"## {title}\n{text[:section_char_limit]}" for title, text in selected_sections
    )[:max_chars]
    return (
        f"Analyzed {len(sections)} main-content sections; returning the {len(selected_sections)} "
        f"most relevant to the question.\n\n{analysis}"
    )