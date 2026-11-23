"""LangGraph entrypoint for the second GAIA agent architecture: supervisor + specialists."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr
import requests

from ..tools.calculator import calculator
from ..tools.exa_search import exa_search
from ..tools.file_tools import find_referenced_file, read_data_file
from ..tools.python_tool import run_python
from ..tools.tavily_extract import tavily_extract
from ..tools.video_analyzer import extract_youtube_url
from ..tools.video_pipeline import analyze_video
from ..tools.web_search import web_search
from ..tools.webpage import read_webpage

TEXTUAL_TOOL_CALL = re.compile(r"^(?P<name>[A-Za-z_]\w*)\[ARGS\](?P<args>\{.*\})$")
FINAL_ANSWER = re.compile(r"<final_answer>\s*(?P<answer>.*?)\s*</final_answer>", re.DOTALL)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
CODE_FENCE = re.compile(r"^```(?:python)?\s*\n(.*?)\n```$", re.DOTALL)
GREETING_PREFIX = re.compile(
    r"^(?:hi|hello|hey|please|can you|could you|would you|i need you to|i want you to)[,:\s]+",
    re.IGNORECASE,
)
TRAILING_FILLER = re.compile(r"[\s.!]*(?:thanks|thank you|please)[.!\s]*$", re.IGNORECASE)

MAX_TEXTUAL_TOOL_CALLS = 8
MAX_ANSWER_ATTEMPTS = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "langgraph_supervisor_agent"


def _read_prompt(file_name: str) -> str:
    """Read one plain-text prompt file for this agent."""
    return (PROMPTS_DIR / file_name).read_text(encoding="utf-8").strip()


INPUT_INTERPRETER_PROMPT = _read_prompt("input_interpreter.txt")
SUPERVISOR_PROMPT = _read_prompt("supervisor.txt")
RESEARCH_AGENT_PROMPT = _read_prompt("research_agent.txt")
RECOVERY_AGENT_PROMPT = _read_prompt("recovery_agent.txt")
CODE_WRITER_PROMPT = _read_prompt("code_writer.txt")
REASONING_PROMPT = _read_prompt("reasoning_agent.txt")
VERIFIER_PROMPT = _read_prompt("verifier_agent.txt")
FINALIZER_PROMPT = _read_prompt("finalizer.txt")

RESEARCH_TOOLS = [web_search, exa_search, read_webpage, tavily_extract]
RECOVERY_TOOLS = [exa_search, web_search, tavily_extract]
REASONING_TOOLS = [run_python, calculator]
VERIFIER_TOOLS = [web_search, calculator]


def _deterministic_clean(text: str) -> str:
    """Strip greetings, filler, and excess whitespace before the LLM interpreter pass."""
    cleaned = " ".join(text.strip().split())
    cleaned = GREETING_PREFIX.sub("", cleaned)
    cleaned = TRAILING_FILLER.sub("", cleaned)
    return cleaned.strip() or text.strip()


def _parse_supervisor_plan(text: str) -> dict[str, Any]:
    """Read the supervisor's JSON decomposition, defaulting to research on malformed output."""
    match = JSON_OBJECT.search(text)
    if not match:
        return {"research": True, "file": False, "code": False, "notes": ""}
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"research": True, "file": False, "code": False, "notes": ""}
    return {
        "research": bool(plan.get("research", False)),
        "file": bool(plan.get("file", False)),
        "code": bool(plan.get("code", False)),
        "notes": str(plan.get("notes", "")),
    }


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence from generated Python code."""
    match = CODE_FENCE.match(text.strip())
    return match.group(1) if match else text.strip()


def _parse_textual_tool_call(content: Any) -> tuple[str, dict[str, Any]] | None:
    """Parse a model's unstructured `tool_name[ARGS]{...}` response format."""
    if not isinstance(content, str):
        return None
    match = TEXTUAL_TOOL_CALL.fullmatch(content.strip())
    if not match:
        return None
    try:
        arguments = json.loads(match.group("args"))
    except json.JSONDecodeError:
        return None
    return (match.group("name"), arguments) if isinstance(arguments, dict) else None


class Agent2State(TypedDict):
    """Shared state exchanged across the supervisor, specialists, and verification loop."""

    question: str
    clarified_question: str
    video_request: bool
    video_evidence: str
    supervisor_plan: dict[str, Any]
    research_finding: str
    file_finding: str
    code_finding: str
    combined_findings: str
    reasoning_output: str
    review: str
    attempts: int
    final_answer: str


class GaiaAgent2:
    """Supervisor-orchestrated GAIA agent with research, file, and code specialists."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        provider = provider or os.environ.get("GAIA_PROVIDER", "auto")
        if provider not in {"auto", "ollama", "huggingface"}:
            raise ValueError("GAIA_PROVIDER must be 'auto', 'ollama', or 'huggingface'.")

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        use_ollama = provider == "ollama" or (
            provider == "auto" and self._ollama_is_available(ollama_url)
        )
        if use_ollama:
            self.provider = "ollama"
            model = ChatOllama(
                model=model_name or os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
                base_url=base_url or ollama_url,
                temperature=0,
            )
        else:
            self.provider = "huggingface"
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "Ollama is unavailable. Set HF_TOKEN to use Hugging Face, or start Ollama."
                )
            model = ChatOpenAI(
                model=model_name or os.environ.get("HF_MODEL", "Qwen/Qwen3-32B"),
                api_key=SecretStr(token),
                base_url=base_url or os.environ.get(
                    "HF_BASE_URL", "https://router.huggingface.co/v1"
                ),
                temperature=0,
            )
        self.model = model
        self.research_solver_graph = create_agent(
            model, RESEARCH_TOOLS, system_prompt=RESEARCH_AGENT_PROMPT
        )
        self.recovery_agent_graph = create_agent(
            model, RECOVERY_TOOLS, system_prompt=RECOVERY_AGENT_PROMPT
        )
        self.reasoning_agent_graph = create_agent(
            model, REASONING_TOOLS, system_prompt=REASONING_PROMPT
        )
        self.verifier_agent_graph = create_agent(
            model, VERIFIER_TOOLS, system_prompt=VERIFIER_PROMPT
        )
        self.research_tools_by_name = {tool.name: tool for tool in RESEARCH_TOOLS}
        self.recovery_tools_by_name = {tool.name: tool for tool in RECOVERY_TOOLS}
        self.reasoning_tools_by_name = {tool.name: tool for tool in REASONING_TOOLS}
        self.verifier_tools_by_name = {tool.name: tool for tool in VERIFIER_TOOLS}
        self.stage_agents = {
            "input_interpreter": create_agent(
                model, [], system_prompt=INPUT_INTERPRETER_PROMPT
            ),
            "supervisor": create_agent(model, [], system_prompt=SUPERVISOR_PROMPT),
            "code_writer": create_agent(model, [], system_prompt=CODE_WRITER_PROMPT),
            "finalizer": create_agent(model, [], system_prompt=FINALIZER_PROMPT),
        }
        self.langfuse_handler = self._create_langfuse_handler()
        self.graph = self._build_workflow()

    @staticmethod
    def _ollama_is_available(base_url: str) -> bool:
        """Return whether an Ollama server responds at the configured endpoint."""
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1)
            return response.ok
        except requests.RequestException:
            return False

    def solve(self, question: str) -> str:
        """Run the full supervisor workflow and return the checked final answer."""
        try:
            result = cast(
                Agent2State,
                self.graph.invoke(
                    cast(Agent2State, {"question": question, "attempts": 0}),
                    config=self._trace_config("gaia2-workflow"),
                ),
            )
            return result.get("final_answer") or result.get("reasoning_output", "")
        except Exception as error:
            return f"Agent error: {error}"
        finally:
            if self.langfuse_handler:
                self.langfuse_handler.client.flush()

    def _build_workflow(self):
        """Build the input-interpreter -> supervisor -> specialists -> verification graph."""
        workflow = StateGraph(Agent2State)
        workflow.add_node("input_interpreter", self._input_interpreter_node)
        workflow.add_node("video_request_checker", self._video_request_checker_node)
        workflow.add_node("video_pipeline", self._video_pipeline_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("research_agent", self._research_agent_node)
        workflow.add_node("file_agent", self._file_agent_node)
        workflow.add_node("code_agent", self._code_agent_node)
        workflow.add_node("combine_findings", self._combine_findings_node)
        workflow.add_node("reasoning_agent", self._reasoning_agent_node)
        workflow.add_node("verifier_agent", self._verifier_agent_node)
        workflow.add_node("finalizer", self._finalizer_node)
        workflow.add_node("recovery_agent", self._recovery_agent_node)

        workflow.add_edge(START, "input_interpreter")
        workflow.add_edge("input_interpreter", "video_request_checker")
        workflow.add_conditional_edges(
            "video_request_checker",
            self._route_by_video_request,
            {"video": "video_pipeline", "supervisor": "supervisor"},
        )
        workflow.add_edge("video_pipeline", "reasoning_agent")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_by_supervisor_plan,
            {
                "research_agent": "research_agent",
                "file_agent": "file_agent",
                "code_agent": "code_agent",
            },
        )
        workflow.add_edge("research_agent", "combine_findings")
        workflow.add_edge("file_agent", "combine_findings")
        workflow.add_edge("code_agent", "combine_findings")
        workflow.add_edge("combine_findings", "reasoning_agent")
        workflow.add_edge("reasoning_agent", "verifier_agent")
        workflow.add_conditional_edges(
            "verifier_agent",
            self._route_after_verifier,
            {"pass": "finalizer", "fail": "recovery_agent"},
        )
        workflow.add_edge("recovery_agent", "supervisor")
        workflow.add_edge("finalizer", END)
        return workflow.compile()

    def _input_interpreter_node(self, state: Agent2State) -> dict[str, str]:
        """Conservatively clean the question: deterministic pass, then a light LLM pass."""
        deterministic = _deterministic_clean(state["question"])
        clarified = self._run_stage_agent(
            "input_interpreter", deterministic, "gaia2-input-interpreter"
        )
        return {"clarified_question": clarified or deterministic}

    def _video_request_checker_node(self, state: Agent2State) -> dict[str, bool]:
        """Identify requests that require visual and audio evidence from a YouTube video."""
        return {"video_request": extract_youtube_url(state["clarified_question"]) is not None}

    @staticmethod
    def _route_by_video_request(state: Agent2State) -> str:
        """Route recognized video requests to the audio/frame analysis pipeline."""
        return "video" if extract_youtube_url(state["clarified_question"]) else "supervisor"

    def _video_pipeline_node(self, state: Agent2State) -> dict[str, str]:
        """Download the video and gather a transcript plus visual observations."""
        url = extract_youtube_url(state["clarified_question"])
        if url is None:
            evidence = "No supported YouTube URL was provided."
        else:
            try:
                evidence = analyze_video.invoke(
                    {"url": url, "question": state["clarified_question"]},
                    config=self._trace_config("gaia2-video-pipeline"),
                )
            except Exception as error:
                evidence = f"Video analysis failed: {error}"
        return {"video_evidence": evidence, "combined_findings": evidence}

    def _supervisor_node(self, state: Agent2State) -> dict[str, dict[str, Any]]:
        """Decompose the task and decide which specialist agents are required."""
        response = self._run_stage_agent(
            "supervisor", state["clarified_question"], "gaia2-supervisor"
        )
        plan = _parse_supervisor_plan(response)
        if not any([plan["research"], plan["file"], plan["code"]]):
            plan["research"] = True
        return {"supervisor_plan": plan}

    @staticmethod
    def _route_by_supervisor_plan(state: Agent2State) -> list[str]:
        """Fan out to every specialist agent the supervisor selected."""
        plan = state["supervisor_plan"]
        targets = [
            name
            for name, key in (
                ("research_agent", "research"),
                ("file_agent", "file"),
                ("code_agent", "code"),
            )
            if plan.get(key)
        ]
        return targets or ["research_agent"]

    def _research_agent_node(self, state: Agent2State) -> dict[str, str]:
        """Gather and verify facts from web sources for the clarified question."""
        finding = self._run_tool_agent(
            self.research_solver_graph,
            self.research_tools_by_name,
            state["clarified_question"],
            "gaia2-research-agent",
        )
        return {"research_finding": finding}

    def _file_agent_node(self, state: Agent2State) -> dict[str, str]:
        """Deterministically inspect a local file referenced in the question, if any."""
        referenced_path = find_referenced_file(state["clarified_question"])
        if referenced_path is None:
            return {"file_finding": "No local file was referenced in the question."}
        try:
            finding = read_data_file.invoke(
                {"path": referenced_path}, config=self._trace_config("gaia2-file-agent")
            )
        except Exception as error:
            finding = f"File reading failed: {error}"
        return {"file_finding": finding}

    def _code_agent_node(self, state: Agent2State) -> dict[str, str]:
        """Write and deterministically execute Python code for the requested computation."""
        code_response = self._run_stage_agent(
            "code_writer",
            (
                f"Task:\n{state['clarified_question']}\n\n"
                "Write Python code that prints only the requested result."
            ),
            "gaia2-code-agent",
        )
        code = _strip_code_fence(code_response)
        try:
            output = run_python.invoke(
                {"code": code}, config=self._trace_config("gaia2-code-execution")
            )
        except Exception as error:
            output = f"Code execution failed: {error}"
        return {"code_finding": f"Code:\n{code}\n\nOutput:\n{output}"}

    def _combine_findings_node(self, state: Agent2State) -> dict[str, str]:
        """Merge the outputs of whichever specialist agents the supervisor selected."""
        parts = []
        if state.get("research_finding"):
            parts.append(f"Research finding:\n{state['research_finding']}")
        if state.get("file_finding"):
            parts.append(f"File finding:\n{state['file_finding']}")
        if state.get("code_finding"):
            parts.append(f"Code finding:\n{state['code_finding']}")
        return {"combined_findings": "\n\n".join(parts) or "No findings were collected."}

    def _reasoning_agent_node(self, state: Agent2State) -> dict[str, str]:
        """Synthesize the collected evidence into a candidate final answer."""
        request = (
            f"Question:\n{state['clarified_question']}\n\n"
            f"Evidence:\n{state['combined_findings']}"
        )
        output = self._run_tool_agent(
            self.reasoning_agent_graph,
            self.reasoning_tools_by_name,
            request,
            "gaia2-reasoning-agent",
        )
        return {"reasoning_output": output}

    def _verifier_agent_node(self, state: Agent2State) -> dict[str, str]:
        """Independently verify the reasoning agent's candidate answer."""
        request = (
            f"Question:\n{state['clarified_question']}\n\n"
            f"Candidate answer:\n{state['reasoning_output']}\n\n"
            "Independently verify this answer using the available tools if needed. Reply "
            "with exactly PASS if correct and well-supported, or FAIL: followed by one "
            "concise reason."
        )
        review = self._run_tool_agent(
            self.verifier_agent_graph,
            self.verifier_tools_by_name,
            request,
            "gaia2-verifier-agent",
        ).strip()
        return {"review": review}

    @staticmethod
    def _route_after_verifier(state: Agent2State) -> str:
        """Finalize on pass or attempt limit; otherwise send the question to recovery."""
        if state["review"].upper().startswith("PASS") or state["attempts"] >= MAX_ANSWER_ATTEMPTS:
            return "pass"
        return "fail"

    def _recovery_agent_node(self, state: Agent2State) -> dict[str, Any]:
        """Search for corrected or additional evidence after a failed verification."""
        request = (
            f"Question:\n{state['clarified_question']}\n\n"
            f"Previous candidate answer:\n{state['reasoning_output']}\n\n"
            f"Verifier feedback:\n{state['review']}\n\n"
            "Find corrected or additional evidence using alternate search queries and page "
            "extraction. Return only the new evidence found."
        )
        recovery_evidence = self._run_tool_agent(
            self.recovery_agent_graph,
            self.recovery_tools_by_name,
            request,
            "gaia2-recovery-agent",
        )
        combined = f"{state.get('combined_findings', '')}\n\nRecovery evidence:\n{recovery_evidence}"
        return {"combined_findings": combined, "attempts": state["attempts"] + 1}

    def _finalizer_node(self, state: Agent2State) -> dict[str, str]:
        """Produce the final answer content with no unrequested reasoning or Markdown."""
        final_answer = self._run_stage_agent(
            "finalizer",
            (
                f"Question:\n{state['clarified_question']}\n\n"
                f"Verified candidate answer:\n{state['reasoning_output']}\n\n"
                f"Verifier notes:\n{state['review']}"
            ),
            "gaia2-finalizer",
        )
        return {"final_answer": final_answer or state["reasoning_output"]}

    def _run_stage_agent(self, name: str, request: str, run_name: str) -> str:
        """Invoke one focused, tool-free ReAct stage agent and return its response text."""
        stage_agent = self.stage_agents.get(name)
        if stage_agent is None:
            try:
                response = self.model.invoke(
                    [{"role": "user", "content": request}], config=self._trace_config(run_name)
                )
            except Exception as error:
                return f"Agent error: {error}"
            return self._extract_final_answer(response.content).strip()
        result = self._safe_invoke(
            stage_agent, {"messages": [{"role": "user", "content": request}]}, run_name
        )
        return self._extract_final_answer(result["messages"][-1].content).strip()

    def _safe_invoke(self, agent_graph: Any, inputs: dict[str, Any], run_name: str) -> dict[str, Any]:
        """Invoke an agent graph, converting transport or native tool-call errors into a safe result.

        Ollama can raise on malformed native tool-call arguments before our tool-level error
        handling ever runs; this keeps such failures from crashing the whole workflow.
        """
        try:
            return agent_graph.invoke(inputs, config=self._trace_config(run_name))
        except Exception as error:
            return {"messages": [AIMessage(content=f"<final_answer>Agent error: {error}</final_answer>")]}

    def _run_tool_agent(
        self, graph: Any, tools_by_name: dict[str, Any], question: str, trace_prefix: str
    ) -> str:
        """Run a ReAct-style agent graph, resolving any textual tool calls it emits."""
        result = self._safe_invoke(
            graph, {"messages": [{"role": "user", "content": question}]}, trace_prefix
        )
        messages = result["messages"]
        for _ in range(MAX_TEXTUAL_TOOL_CALLS):
            content = messages[-1].content
            tool_call = _parse_textual_tool_call(content)
            if tool_call is None:
                return self._extract_final_answer(content)

            tool_name, arguments = tool_call
            tool = tools_by_name.get(tool_name)
            if tool is None:
                return f"Unsupported tool requested by model: {tool_name}"
            try:
                tool_result = tool.invoke(
                    arguments, config=self._trace_config(f"{trace_prefix}-tool")
                )
            except Exception as error:
                tool_result = f"Tool execution failed: {error}"

            result = self._safe_invoke(
                graph,
                {
                    "messages": [
                        *messages,
                        {"role": "user", "content": f"Tool result for {tool_name}:\n{tool_result}"},
                    ]
                },
                trace_prefix,
            )
            messages = result["messages"]
        result = self._safe_invoke(
            graph,
            {
                "messages": [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Tool-call limit reached. Do not call any more tools. Based only on "
                            "the evidence already collected, return the requested final answer now."
                        ),
                    },
                ]
            },
            trace_prefix,
        )
        return self._extract_final_answer(result["messages"][-1].content)

    def _trace_config(self, run_name: str = "gaia2-agent") -> RunnableConfig:
        """Return callback configuration for one agent invocation."""
        if not self.langfuse_handler:
            return {}
        return {
            "callbacks": [self.langfuse_handler],
            "run_name": run_name,
            "metadata": {"provider": self.provider},
        }

    @staticmethod
    def _create_langfuse_handler() -> CallbackHandler | None:
        """Create a Langfuse handler only when its credentials are configured."""
        if not (
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
        ):
            return None
        return CallbackHandler()

    @staticmethod
    def _extract_final_answer(content: Any) -> str:
        """Return the delimited benchmark answer, or the model response as a fallback."""
        response = content if isinstance(content, str) else str(content)
        match = FINAL_ANSWER.search(response)
        if match:
            return match.group("answer").strip()
        return response


def build_agent_2() -> GaiaAgent2:
    """Create a GaiaAgent2 using the configured local Ollama model."""
    return GaiaAgent2()


def solve_question_2(question: str) -> str:
    """Return the answer for one question using the default GaiaAgent2."""
    return build_agent_2().solve(question)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer a GAIA-style question (agent 2).")
    parser.add_argument("question", help="Question for the agent to solve")
    args = parser.parse_args()
    print(solve_question_2(args.question))
