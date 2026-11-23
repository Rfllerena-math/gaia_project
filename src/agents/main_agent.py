"""Improved single-agent GAIA solver using only Ollama or OpenAI."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypedDict, cast

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr

from ..tools.attachment_reader import inspect_attachment
from ..tools.calculator import calculator
from ..tools.python_tool import run_python
from ..tools.video_pipeline import analyze_video
from ..tools_for_agent4 import research_web, tavily_extract, tavily_search, verify_answer

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "main_agent"

SUPPORTED_PROVIDERS = frozenset({"auto", "ollama", "openai"})
COMMON_ENGLISH_WORDS = frozenset(
    {
        "a", "an", "and", "are", "calculate", "can", "do", "find", "for",
        "from", "how", "i", "in", "is", "of", "please", "the", "to", "use",
        "what", "when", "where", "which", "who", "with", "you",
    }
)
WORD_PATTERN = re.compile(r"[a-z]+")
NUMERIC_REQUEST = re.compile(
    r"\b(?:how many|count|number of|total|sum|difference|percentage|percent|average)\b",
    re.IGNORECASE,
)
DIRECT_NUMERIC_ANSWER = re.compile(
    r"^\s*-?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|[A-Za-z][A-Za-z -]*))?[.!]?\s*$"
)
MAX_RECOVERY_EVIDENCE_CHARS = 20_000

SYSTEM_PROMPT = (PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")

TOOLS = [
    research_web,
    tavily_search,
    tavily_extract,
    run_python,
    calculator,
    analyze_video,
    inspect_attachment,
    verify_answer,
]


class AgentTrace(TypedDict):
    """Visible execution details from one run."""

    answer: str
    steps: list[str]


def _english_word_score(text: str) -> int:
    return sum(word in COMMON_ENGLISH_WORDS for word in WORD_PATTERN.findall(text.lower()))


def normalize_user_prompt(prompt: str) -> str:
    """Restore a character-reversed English prompt while preserving normal text."""
    reversed_prompt = prompt[::-1]
    if _english_word_score(reversed_prompt) >= 2 and (
        _english_word_score(reversed_prompt) > _english_word_score(prompt)
    ):
        return reversed_prompt
    return prompt


def _requires_direct_numeric_answer(question: str) -> bool:
    return NUMERIC_REQUEST.search(question) is not None


def _is_direct_numeric_answer(answer: str) -> bool:
    return DIRECT_NUMERIC_ANSWER.fullmatch(answer) is not None


class GaiaAgent4:
    """A single tool-enabled agent backed by Ollama or OpenAI."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        requested_provider = (provider or os.environ.get("GAIA_PROVIDER", "auto")).lower()
        if requested_provider not in SUPPORTED_PROVIDERS:
            choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(f"GAIA_PROVIDER must be one of: {choices}.")

        ollama_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        if requested_provider == "auto":
            requested_provider = (
                "ollama" if self._ollama_is_available(ollama_url) else "openai"
            )

        self.provider = requested_provider
        self.model_name = (model_name or self._default_model(requested_provider)).strip()
        if not self.model_name:
            raise ValueError("The model name must not be empty.")
        self.langfuse_handler = self._create_langfuse_handler()
        self._execution_steps: list[str] = []

        if self.provider == "ollama":
            self.model = ChatOllama(
                model=self.model_name,
                base_url=ollama_url,
                temperature=0,
                num_ctx=self._positive_env_int("OLLAMA_NUM_CTX", 32768),
                num_predict=self._positive_env_int("OLLAMA_NUM_PREDICT", 4096),
                reasoning=self._ollama_reasoning(self.model_name),
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OpenAI was selected but OPENAI_API_KEY is not set. Set it, start Ollama, "
                    "or choose provider='ollama'."
                )
            self.model = ChatOpenAI(
                model=self.model_name,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0,
            )

        self.agent = create_agent(self.model, TOOLS, system_prompt=SYSTEM_PROMPT)

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "ollama":
            return os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
        return os.environ.get("OPENAI_MODEL", "gpt-5-mini")

    @staticmethod
    def _positive_env_int(name: str, default: int) -> int:
        raw_value = os.environ.get(name, str(default))
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"{name} must be a positive integer.") from error
        if value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    @staticmethod
    def _ollama_reasoning(model_name: str) -> bool | str | None:
        configured = os.environ.get("OLLAMA_REASONING")
        if configured is None:
            return "low" if model_name.lower().startswith("gpt-oss") else None
        normalized = configured.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
        return normalized or None

    @staticmethod
    def _ollama_is_available(base_url: str) -> bool:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1)
            return response.ok
        except requests.RequestException:
            return False

    def solve(self, question: str) -> str:
        """Solve one question and return only the final answer."""
        prompt = normalize_user_prompt(question.strip())
        if not prompt:
            raise ValueError("question must not be empty")

        steps = [f"Input: {prompt}", f"Provider: {self.provider}"]
        try:
            try:
                result = self._invoke([{"role": "user", "content": prompt}], "gaia4-agent")
            except Exception as error:
                if not self._is_malformed_tool_call_error(error):
                    raise
                steps.append(
                    "Recovery: Ollama produced malformed tool arguments; "
                    "researching deterministically and finalizing without tools."
                )
                answer = self._recover_malformed_tool_call(prompt, error)
                result = {"messages": []}
            else:
                steps.extend(self._react_execution_steps(result))
                try:
                    answer = self._extract_response(result)
                except RuntimeError as error:
                    if str(error) != "The model returned an empty answer.":
                        raise
                    steps.append(
                        "Recovery: final model message was empty; "
                        "finalizing from retrieved evidence."
                    )
                    answer = self._recover_empty_answer(prompt, result)

            if _requires_direct_numeric_answer(prompt) and not _is_direct_numeric_answer(answer):
                steps.append("Validation: answer is not a direct numeric result; retrying.")
                correction = (
                    "Your previous response did not directly answer this quantitative question: "
                    f"{prompt}\nUse the available research and calculation tools as needed, then "
                    "return only the requested numeric value (and unit only if requested)."
                )
                result = self._invoke(
                    [*result.get("messages", []), {"role": "user", "content": correction}],
                    "gaia4-numeric-correction",
                )
                answer = self._extract_response(result)
                steps.extend(self._react_execution_steps(result))

            steps.append(f"Answer: {answer}")
            self._execution_steps = steps
            return answer
        except Exception as error:
            self._execution_steps = [*steps, f"Error: {error}"]
            raise RuntimeError(f"Agent failed using {self.provider}: {error}") from error
        finally:
            if self.langfuse_handler:
                self.langfuse_handler.client.flush()

    def _invoke(self, messages: list[Any], run_name: str) -> dict[str, Any]:
        try:
            return cast(
                dict[str, Any],
                self.agent.invoke(
                    {"messages": messages},
                    config=self._trace_config(run_name),
                ),
            )
        except Exception as error:
            if not self._is_malformed_tool_call_error(error):
                raise
            repair_instruction = (
                "Retry the task from the beginning. Your prior tool call was rejected because "
                "its arguments were not valid JSON. Every tool requires a JSON object. Examples: "
                "run_python uses {\"code\": \"print(2 + 2)\"}; calculator uses "
                "{\"expression\": \"2 + 2\"}. Never send raw code or plain text as arguments."
            )
            return cast(
                dict[str, Any],
                self.agent.invoke(
                    {
                        "messages": [
                            *messages,
                            {"role": "user", "content": repair_instruction},
                        ]
                    },
                    config=self._trace_config(f"{run_name}-json-repair"),
                ),
            )

    @staticmethod
    def _is_malformed_tool_call_error(error: Exception) -> bool:
        message = str(error).lower()
        return "error parsing tool call" in message or (
            "tool" in message and "arguments" in message and "not valid json" in message
        )

    def _recover_malformed_tool_call(self, prompt: str, original_error: Exception) -> str:
        """Bypass native tool calling after two malformed Ollama generations."""
        evidence = research_web.invoke({"question": prompt, "max_sources": 3})
        if evidence.startswith(("Web research failed:", "Web research found no usable")):
            raise RuntimeError(
                f"Ollama produced invalid tool JSON and fallback research failed: {evidence}"
            ) from original_error

        response = self.model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the supplied web evidence. Do not call "
                        "tools. Return exactly the requested answer without a preamble. For a "
                        "count, return only the number."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{prompt}\n\nWeb evidence:\n{evidence}",
                },
            ],
            config=self._trace_config("gaia4-malformed-tool-recovery"),
        )
        answer = self._message_text(response)
        if not answer:
            raise RuntimeError(
                "Ollama returned no answer during malformed-tool recovery. "
                "Use a tool-capable Ollama model other than gpt-oss or select OpenAI."
            ) from original_error
        return answer

    def _recover_empty_answer(self, prompt: str, result: dict[str, Any]) -> str:
        """Request a final answer without tools when a reasoning model emits no final text."""
        evidence_parts: list[str] = []
        remaining = MAX_RECOVERY_EVIDENCE_CHARS
        for message in result.get("messages", []):
            if not getattr(message, "name", None):
                continue
            content = self._message_text(message)
            if not content:
                continue
            excerpt = content[:remaining]
            evidence_parts.append(f"{getattr(message, 'name', 'tool')}:\n{excerpt}")
            remaining -= len(excerpt)
            if remaining <= 0:
                break

        evidence = "\n\n".join(evidence_parts) or "No tool evidence was returned."
        finalizer_prompt = (
            f"Question:\n{prompt}\n\n"
            f"Retrieved tool evidence:\n{evidence}\n\n"
            "Give the exact final answer now. Do not call tools, describe your reasoning, or add "
            "a preamble. If the question asks for a count, return only the number."
        )
        response = self.model.invoke(
            [
                {
                    "role": "system",
                    "content": "You finalize answers from supplied evidence concisely.",
                },
                {"role": "user", "content": finalizer_prompt},
            ],
            config=self._trace_config("gaia4-empty-answer-recovery"),
        )
        answer = self._message_text(response)
        if not answer:
            metadata = getattr(response, "response_metadata", {}) or {}
            done_reason = metadata.get("done_reason") or metadata.get("finish_reason") or "unknown"
            raise RuntimeError(
                "The model returned an empty answer after recovery "
                f"(finish reason: {done_reason}). Try a larger OLLAMA_NUM_CTX or another model."
            )
        return answer

    def solve_with_trace(self, question: str) -> AgentTrace:
        answer = self.solve(question)
        return {"answer": answer, "steps": self.get_reasoning_steps()}

    def _trace_config(self, run_name: str) -> RunnableConfig:
        if not self.langfuse_handler:
            return {}
        return {
            "callbacks": [self.langfuse_handler],
            "run_name": run_name,
            "metadata": {"provider": self.provider, "model": self.model_name},
        }

    @staticmethod
    def _create_langfuse_handler() -> CallbackHandler | None:
        if not (
            os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        ):
            return None
        return CallbackHandler()

    @staticmethod
    def _extract_response(result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        if not messages:
            raise RuntimeError("The model returned no messages.")
        answer = GaiaAgent4._message_text(messages[-1])
        if not answer:
            raise RuntimeError("The model returned an empty answer.")
        return answer

    @staticmethod
    def _message_text(message: Any) -> str:
        """Extract visible answer text from string or content-block messages."""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content).strip() if content else ""

        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {
                "text",
                "output_text",
            }:
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(part.strip() for part in text_parts if part.strip()).strip()

    def get_reasoning_steps(self) -> list[str]:
        """Return observable execution events, never private chain-of-thought."""
        return list(self._execution_steps)

    @staticmethod
    def _react_execution_steps(result: dict[str, Any]) -> list[str]:
        steps: list[str] = []
        for message in result.get("messages", []):
            for tool_call in getattr(message, "tool_calls", []) or []:
                steps.append(
                    f"Tool call: {tool_call.get('name', 'unknown')}"
                    f"({tool_call.get('args', {})})"
                )
            tool_name = getattr(message, "name", None)
            if tool_name:
                content = str(getattr(message, "content", "")).strip()
                steps.append(f"Tool result: {tool_name}: {content[:1000]}")
        return steps


def build_agent_4() -> GaiaAgent4:
    return GaiaAgent4()


def solve_question_4(question: str) -> str:
    return build_agent_4().solve(question)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer a GAIA question with Ollama or OpenAI.")
    parser.add_argument("question", help="Question for the agent to solve")
    parser.add_argument("--provider", choices=sorted(SUPPORTED_PROVIDERS), default=None)
    parser.add_argument("--model", default=None, help="Override the provider's default model")
    parser.add_argument("--base-url", default=None, help="Override the provider API URL")
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    agent = GaiaAgent4(model_name=args.model, base_url=args.base_url, provider=args.provider)
    trace = agent.solve_with_trace(args.question)
    print(trace["answer"])
    if args.show_trace:
        print("\nVisible execution trace:")
        print("\n".join(trace["steps"]))
