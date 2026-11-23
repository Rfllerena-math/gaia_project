# GAIA Project

Two Python agent architectures built with [LangChain](https://python.langchain.com/) and
[LangGraph](https://langchain-ai.github.io/langgraph/) to answer
[GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard)-style questions, plus
the Gradio app used to run and submit answers for the Hugging Face Agents Course evaluation.

## Agents

### `main_agent.py` — single tool-calling agent (used by `app.py`)

[`src/agents/main_agent.py`](src/agents/main_agent.py) defines `GaiaAgent4`, a single
LangChain `create_agent` ReAct loop backed by either a local Ollama model or OpenAI. It has
built-in recovery for malformed tool calls and empty model responses, and a numeric-answer
validation/retry pass for counting-style questions. This is the agent wired into
[`app.py`](app.py), the Gradio evaluation-runner UI.

Tools: web research via Tavily (`research_web`, `tavily_search`, `tavily_extract`,
`verify_answer`), `run_python`, `calculator`, `analyze_video`, and `inspect_attachment` (local
file/image/audio inspection).

### `langgraph_supervisor_agent.py` — supervisor + specialist sub-agents

[`src/agents/langgraph_supervisor_agent.py`](src/agents/langgraph_supervisor_agent.py) defines
`GaiaAgent2`, a multi-agent architecture orchestrated as a LangGraph `StateGraph`. An
input-interpreter node cleans the question, a supervisor node decomposes it and decides which
specialist sub-agents to invoke (research, file, code), their findings are combined and passed to
a reasoning agent, an independent verifier agent checks the candidate answer, and a recovery
agent gathers additional evidence and loops back through the supervisor on verification failure.

```
START -> input_interpreter -> video_request_checker -> [video_pipeline | supervisor]
                                                              |
                               +------------------------------+------------------------------+
                               v                              v                              v
                        research_agent                   file_agent                     code_agent
                               \_______________________________|______________________________/
                                                              v
                                                    combine_findings -> reasoning_agent -> verifier_agent
                                                                                                |
                                                                              pass -> finalizer -> END
                                                                              fail -> recovery_agent -> supervisor
```

Tools: `web_search`, `exa_search`, `read_webpage`, `tavily_extract`, `run_python`, `calculator`,
`analyze_video`.

## Project layout

```
gaia_project/
├── app.py                              # Gradio UI that runs main_agent against the GAIA evaluation API
├── src/
│   ├── agents/
│   │   ├── main_agent.py               # GaiaAgent4 - single tool-calling agent (used by app.py)
│   │   └── langgraph_supervisor_agent.py  # GaiaAgent2 - LangGraph supervisor + specialist sub-agents
│   ├── tools/                          # Shared LangChain tools used by both agents
│   └── tools_for_agent4/               # Tavily-first research tools used only by main_agent
├── prompts/                            # Plain-text system prompts, one file per agent role
│   ├── main_agent/
│   │   └── system_prompt.txt
│   └── langgraph_supervisor_agent/
│       ├── input_interpreter.txt
│       ├── supervisor.txt
│       ├── research_agent.txt
│       ├── recovery_agent.txt
│       ├── code_writer.txt
│       ├── reasoning_agent.txt
│       ├── verifier_agent.txt
│       └── finalizer.txt
├── tests/                              # Unit tests for both agents and their tools
├── requirements.txt
└── .env.example
```

Prompts are plain `.txt` files (no Jinja2/templating) read at import time with
`Path.read_text()`, since none of them interpolate runtime variables — editing a prompt is just
editing its text file, no code change required.

## Setup

```bash
git clone https://github.com/Rfllerena-math/gaia_project.git
cd gaia_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your own API keys
```

By default both agents use a local Ollama server when one is available at
`http://localhost:11434`; otherwise `main_agent` falls back to OpenAI and
`langgraph_supervisor_agent` falls back to Hugging Face Inference Providers. Set `GAIA_PROVIDER`
to override automatic selection.

Research tools require a `TAVILY_API_KEY` (both agents) and, for
`langgraph_supervisor_agent`, optionally `EXA_API_KEY`/`BRAVE_SEARCH_API_KEY` for its
`exa_search`/`web_search` fallbacks. Langfuse tracing is enabled automatically when
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set.

## Running

```bash
# Single tool-calling agent
python -m src.agents.main_agent "What is the capital of France?"

# LangGraph supervisor + specialists
python -m src.agents.langgraph_supervisor_agent "What is the capital of France?"

# Gradio evaluation-runner app (what main_agent is wired into)
python app.py
```

## Testing

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
