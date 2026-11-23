import importlib
import os
import unittest
from unittest.mock import patch


class Agent4TavilyToolTests(unittest.TestCase):
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_search_uses_advanced_tavily_without_generated_answer(self) -> None:
        client = importlib.import_module("src.tools_for_agent4._tavily_client")
        with patch.object(client.requests, "post") as post:
            post.return_value.json.return_value = {"results": []}
            post.return_value.raise_for_status.return_value = None

            client.search_tavily("test question", max_results=8)

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["search_depth"], "advanced")
        self.assertEqual(payload["chunks_per_source"], 3)
        self.assertFalse(payload["include_answer"])
        self.assertFalse(payload["include_raw_content"])

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_extract_accepts_multiple_urls_and_question(self) -> None:
        client = importlib.import_module("src.tools_for_agent4._tavily_client")
        with patch.object(client.requests, "post") as post:
            post.return_value.json.return_value = {"results": [], "failed_results": []}
            post.return_value.raise_for_status.return_value = None

            client.extract_tavily(
                ["https://example.com/a", "https://example.com/b"],
                query="What is relevant?",
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(len(payload["urls"]), 2)
        self.assertEqual(payload["query"], "What is relevant?")
        self.assertEqual(payload["chunks_per_source"], 5)
        self.assertEqual(payload["extract_depth"], "advanced")

    def test_research_web_searches_then_extracts_ranked_sources(self) -> None:
        module = importlib.import_module("src.tools_for_agent4.research_web")
        search_results = [
            {"title": "Lower", "url": "https://example.com/low", "score": 0.4},
            {"title": "Higher", "url": "https://example.com/high", "score": 0.9},
        ]
        extracted = [
            {"url": "https://example.com/high", "raw_content": "Relevant evidence"}
        ]
        with (
            patch.object(module, "search_tavily", return_value=search_results),
            patch.object(module, "extract_tavily", return_value=(extracted, [])) as extract,
        ):
            result = module.research_web.invoke(
                {"question": "What happened?", "max_sources": 1}
            )

        self.assertIn("Relevant evidence", result)
        self.assertIn("https://example.com/high", result)
        self.assertEqual(extract.call_args.args[0], ["https://example.com/high"])
        self.assertEqual(extract.call_args.kwargs["query"], "What happened?")


if __name__ == "__main__":
    unittest.main()
