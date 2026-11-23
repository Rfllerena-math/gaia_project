import unittest
from unittest.mock import patch

from src.tools.web_search import _search_duckduckgo, web_search


class DuckDuckGoSearchTests(unittest.TestCase):
    @patch("src.tools.web_search.DDGS")
    def test_search_formats_titles_urls_and_complete_snippets(self, mock_ddgs) -> None:
        mock_ddgs.return_value.text.return_value = [
            {
                "title": "Mercedes Sosa discography",
                "href": "https://example.com/discography",
                "body": "A complete discography with studio album publication years.",
            }
        ]

        result = _search_duckduckgo("Mercedes Sosa studio albums")

        self.assertIn("Mercedes Sosa discography", result)
        self.assertIn("https://example.com/discography", result)
        self.assertIn("complete discography with studio album publication years", result)
        mock_ddgs.return_value.text.assert_called_once_with(
            "Mercedes Sosa studio albums", max_results=10
        )

    @patch("src.tools.web_search._search_duckduckgo", side_effect=RuntimeError("offline"))
    def test_search_reports_duckduckgo_failure_without_brave_key(self, mock_search) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = web_search.invoke({"query": "test query"})

        self.assertIn("DuckDuckGo: offline", result)
        self.assertIn("Brave: BRAVE_SEARCH_API_KEY is not configured", result)
        mock_search.assert_called_once_with("test query")


if __name__ == "__main__":
    unittest.main()