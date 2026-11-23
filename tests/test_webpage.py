import unittest
from unittest.mock import patch

from src.tools.webpage import analyze_webpage, read_webpage


class WebpageReaderTests(unittest.TestCase):
    @patch("src.tools.webpage.requests.get")
    def test_read_webpage_extracts_wikipedia_article_not_navigation(self, mock_get) -> None:
        mock_get.return_value.text = """
            <html><body>
              <nav>Navigation item</nav>
              <div id="mw-content-text"><div class="mw-parser-output">
                <p>Introduction</p><h2>Discography</h2><p>Later album details</p>
              </div></div>
            </body></html>
        """
        mock_get.return_value.raise_for_status.return_value = None

        result = read_webpage.invoke({"url": "https://en.wikipedia.org/wiki/Test"})

        self.assertIn("Introduction", result)
        self.assertIn("Discography", result)
        self.assertIn("Later album details", result)
        self.assertNotIn("Navigation item", result)
        self.assertIn("Content characters 0-", result)

    @patch("src.tools.webpage.requests.get")
    def test_read_webpage_can_return_later_content_chunk(self, mock_get) -> None:
        mock_get.return_value.text = "<html><main>abcdefghij</main></html>"
        mock_get.return_value.raise_for_status.return_value = None

        result = read_webpage.invoke(
            {"url": "https://example.com", "start_char": 4, "max_chars": 3}
        )

        self.assertIn("Content characters 4-7 of 10", result)
        self.assertTrue(result.endswith("efg"))

    @patch("src.tools.webpage.requests.get")
    def test_analyze_webpage_scans_later_relevant_sections(self, mock_get) -> None:
        mock_get.return_value.text = """
            <html><main>
              <h1>Mercedes Sosa</h1><p>Biographical introduction.</p>
              <h2>Discography</h2><p>Studio albums published in 2000 and 2007.</p>
              <h2>References</h2><p>Source notes.</p>
            </main></html>
        """
        mock_get.return_value.raise_for_status.return_value = None

        result = analyze_webpage.invoke(
            {"url": "https://example.com", "question": "Which studio albums were published in 2007?"}
        )

        self.assertIn("Analyzed 3 main-content sections", result)
        self.assertIn("## Discography", result)
        self.assertIn("Studio albums published in 2000 and 2007.", result)

    @patch("src.tools.webpage.requests.get")
    def test_analyze_webpage_keeps_every_selected_section_in_response(self, mock_get) -> None:
        mock_get.return_value.text = """
            <html><main>
              <h2>Life</h2><p>2000 2001 2002 2003 2004 2005 2006 2007 2008 2009</p>
              <h2>Discography</h2><p>Studio albums list</p>
            </main></html>
        """
        mock_get.return_value.raise_for_status.return_value = None

        result = analyze_webpage.invoke(
            {
                "url": "https://example.com",
                "question": "Which albums were published in 2000?",
                "max_sections": 2,
                "max_chars": 80,
            }
        )

        self.assertIn("## Discography", result)


if __name__ == "__main__":
    unittest.main()