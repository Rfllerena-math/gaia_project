import os
import unittest
from typing import Any, cast
from unittest.mock import patch

from src.agents.main_agent import GaiaAgent4, TOOLS, normalize_user_prompt
from src.tools.attachment_reader import inspect_attachment


class FakeAgent:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = answers or ["Paris"]
        self.inputs: list[dict[str, object]] = []

    def invoke(self, inputs: dict[str, object], **kwargs: object) -> dict[str, object]:
        self.inputs.append(inputs)
        answer = self.answers[min(len(self.inputs) - 1, len(self.answers) - 1)]
        return {"messages": [type("Message", (), {"content": answer})()]}


class FakeFinalizerModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.inputs: list[object] = []

    def invoke(self, inputs: object, **kwargs: object) -> object:
        self.inputs.append(inputs)
        return type("Message", (), {"content": self.answer, "response_metadata": {}})()


class GaiaAgent4Tests(unittest.TestCase):
    def test_normalizes_reversed_prompt(self) -> None:
        self.assertEqual(
            normalize_user_prompt("?ecnarF fo latipac eht si tahW"),
            "What is the capital of France?",
        )

    def test_tools_include_attachment_reader(self) -> None:
        self.assertIn(inspect_attachment, TOOLS)

    def test_agent_uses_only_the_agent4_tavily_web_stack(self) -> None:
        tool_names = {tool.name for tool in TOOLS}

        self.assertTrue({"research_web", "tavily_search", "tavily_extract"} <= tool_names)
        self.assertNotIn("web_search", tool_names)
        self.assertNotIn("exa_search", tool_names)

    def test_only_supported_providers_are_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "GAIA_PROVIDER"):
            GaiaAgent4(provider="gemini")

    @patch("src.agents.main_agent.GaiaAgent4._create_langfuse_handler", return_value=None)
    @patch("src.agents.main_agent.create_agent")
    @patch("src.agents.main_agent.ChatOllama")
    def test_ollama_configuration_supports_long_research_context(
        self, chat_ollama: Any, create_agent: Any, handler: Any
    ) -> None:
        agent = GaiaAgent4(provider="ollama", model_name="gpt-oss:20b ")

        self.assertEqual(agent.model_name, "gpt-oss:20b")
        kwargs = chat_ollama.call_args.kwargs
        self.assertEqual(kwargs["num_ctx"], 32768)
        self.assertEqual(kwargs["num_predict"], 4096)
        self.assertEqual(kwargs["reasoning"], "low")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("src.agents.main_agent.create_agent")
    @patch("src.agents.main_agent.ChatOpenAI")
    def test_openai_uses_same_tool_enabled_agent(self, chat_openai: Any, create_agent: Any) -> None:
        GaiaAgent4(provider="openai", model_name="test-model")

        chat_openai.assert_called_once()
        create_agent.assert_called_once()
        self.assertEqual(create_agent.call_args.args[1], TOOLS)

    def test_solve_passes_normalized_prompt(self) -> None:
        agent = GaiaAgent4.__new__(GaiaAgent4)
        fake = FakeAgent()
        agent.provider = "ollama"
        agent.model_name = "test"
        agent.agent = cast(Any, fake)
        agent.langfuse_handler = None
        agent._execution_steps = []

        self.assertEqual(agent.solve("?ecnarF fo latipac eht si tahW"), "Paris")
        self.assertEqual(
            fake.inputs[0],
            {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
        )

    def test_numeric_answer_is_retried(self) -> None:
        agent = GaiaAgent4.__new__(GaiaAgent4)
        fake = FakeAgent(["There were several.", "5"])
        agent.provider = "ollama"
        agent.model_name = "test"
        agent.agent = cast(Any, fake)
        agent.langfuse_handler = None
        agent._execution_steps = []

        trace = agent.solve_with_trace("How many albums were published?")

        self.assertEqual(trace["answer"], "5")
        self.assertEqual(len(fake.inputs), 2)
        self.assertIn("Validation: answer is not a direct numeric result; retrying.", trace["steps"])

    def test_empty_ollama_content_is_finalized_from_tool_evidence(self) -> None:
        class EmptyAnswerAgent:
            def invoke(self, inputs: object, **kwargs: object) -> dict[str, object]:
                return {
                    "messages": [
                        type(
                            "ToolResult",
                            (),
                            {"name": "research_web", "content": "Six qualifying albums."},
                        )(),
                        type("Message", (), {"content": ""})(),
                    ]
                }

        agent = GaiaAgent4.__new__(GaiaAgent4)
        finalizer = FakeFinalizerModel("6")
        agent.provider = "ollama"
        agent.model_name = "gpt-oss:20b"
        agent.agent = cast(Any, EmptyAnswerAgent())
        agent.model = cast(Any, finalizer)
        agent.langfuse_handler = None
        agent._execution_steps = []

        trace = agent.solve_with_trace("How many albums were published?")

        self.assertEqual(trace["answer"], "6")
        self.assertIn(
            "Recovery: final model message was empty; finalizing from retrieved evidence.",
            trace["steps"],
        )
        self.assertEqual(len(finalizer.inputs), 1)

    def test_malformed_ollama_tool_call_is_retried_with_json_instruction(self) -> None:
        class RepairableAgent:
            def __init__(self) -> None:
                self.inputs: list[dict[str, object]] = []

            def invoke(self, inputs: dict[str, object], **kwargs: object) -> dict[str, object]:
                self.inputs.append(inputs)
                if len(self.inputs) == 1:
                    raise RuntimeError(
                        "error parsing tool call: raw='print(2 + 2)', "
                        "err=invalid character 'p' looking for beginning of value"
                    )
                return {"messages": [type("Message", (), {"content": "4"})()]}

        agent = GaiaAgent4.__new__(GaiaAgent4)
        repairable = RepairableAgent()
        agent.provider = "ollama"
        agent.model_name = "gpt-oss:20b"
        agent.agent = cast(Any, repairable)
        agent.langfuse_handler = None
        agent._execution_steps = []

        self.assertEqual(agent.solve("What is 2 + 2?"), "4")
        self.assertEqual(len(repairable.inputs), 2)
        retry_messages = cast(list[object], repairable.inputs[1]["messages"])
        self.assertIn("valid JSON", str(retry_messages[-1]))

    def test_repeated_malformed_tool_call_uses_research_without_native_tools(self) -> None:
        class BrokenAgent:
            def invoke(self, inputs: object, **kwargs: object) -> dict[str, object]:
                raise RuntimeError(
                    "error parsing tool call: raw='import itertools', "
                    "err=invalid character 'i' looking for beginning of value"
                )

        agent = GaiaAgent4.__new__(GaiaAgent4)
        finalizer = FakeFinalizerModel("6")
        agent.provider = "ollama"
        agent.model_name = "gpt-oss:20b"
        agent.agent = cast(Any, BrokenAgent())
        agent.model = cast(Any, finalizer)
        agent.langfuse_handler = None
        agent._execution_steps = []

        with patch("src.agents.main_agent.research_web") as research:
            research.invoke.return_value = (
                "SOURCE 1\nSix studio albums match the date range."
            )
            trace = agent.solve_with_trace("How many studio albums match?")

        self.assertEqual(trace["answer"], "6")
        self.assertIn(
            "Recovery: Ollama produced malformed tool arguments; "
            "researching deterministically and finalizing without tools.",
            trace["steps"],
        )
        self.assertEqual(len(finalizer.inputs), 1)

    def test_extract_response_supports_text_content_blocks(self) -> None:
        result = {
            "messages": [
                type(
                    "Message",
                    (),
                    {"content": [{"type": "text", "text": "Paris"}]},
                )()
            ]
        }

        self.assertEqual(GaiaAgent4._extract_response(result), "Paris")

    def test_empty_question_is_rejected(self) -> None:
        agent = GaiaAgent4.__new__(GaiaAgent4)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            agent.solve("   ")


if __name__ == "__main__":
    unittest.main()
