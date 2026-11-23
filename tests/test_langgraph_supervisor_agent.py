import unittest
from types import SimpleNamespace

from src.agents.langgraph_supervisor_agent import GaiaAgent2, _deterministic_clean, _parse_supervisor_plan, _strip_code_fence


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeStageAgent:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def invoke(self, *args, **kwargs) -> dict[str, list[SimpleNamespace]]:
        return {"messages": [SimpleNamespace(content=next(self.responses))]}


class Agent2WorkflowNodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = GaiaAgent2.__new__(GaiaAgent2)
        self.agent.langfuse_handler = None

    def test_deterministic_clean_removes_greeting_and_filler(self) -> None:
        cleaned = _deterministic_clean(
            "  hey tell me the capital of France, thanks  "
        )

        self.assertEqual(cleaned, "tell me the capital of France,")

    def test_supervisor_plan_parses_json_and_defaults_to_research(self) -> None:
        plan = _parse_supervisor_plan('{"research": true, "file": false, "code": false}')
        self.assertTrue(plan["research"])

        fallback_plan = _parse_supervisor_plan("not json")
        self.assertTrue(fallback_plan["research"])

    def test_route_by_supervisor_plan_selects_every_matching_specialist(self) -> None:
        state = {
            "supervisor_plan": {"research": True, "file": True, "code": False}
        }

        targets = self.agent._route_by_supervisor_plan(state)

        self.assertEqual(set(targets), {"research_agent", "file_agent"})

    def test_route_by_supervisor_plan_defaults_to_research_when_empty(self) -> None:
        state = {"supervisor_plan": {"research": False, "file": False, "code": False}}

        targets = self.agent._route_by_supervisor_plan(state)

        self.assertEqual(targets, ["research_agent"])

    def test_route_by_video_request_detects_youtube_url(self) -> None:
        state = {"clarified_question": "Analyze https://youtu.be/example for birds."}

        self.assertEqual(self.agent._route_by_video_request(state), "video")

    def test_strip_code_fence_removes_markdown_wrapper(self) -> None:
        fenced = "```python\nprint(1 + 1)\n```"

        self.assertEqual(_strip_code_fence(fenced), "print(1 + 1)")

    def test_combine_findings_merges_available_sections(self) -> None:
        result = self.agent._combine_findings_node(
            {
                "research_finding": "Paris is the capital of France.",
                "file_finding": "",
                "code_finding": "",
            }
        )

        self.assertIn("Research finding:", result["combined_findings"])
        self.assertNotIn("File finding:", result["combined_findings"])

    def test_file_agent_reports_when_no_file_is_referenced(self) -> None:
        result = self.agent._file_agent_node({"clarified_question": "What is the capital of Spain?"})

        self.assertEqual(
            result["file_finding"], "No local file was referenced in the question."
        )

    def test_route_after_verifier_passes_on_approval(self) -> None:
        state = {"review": "PASS", "attempts": 0}

        self.assertEqual(self.agent._route_after_verifier(state), "pass")

    def test_route_after_verifier_fails_then_forces_pass_at_attempt_limit(self) -> None:
        failing_state = {"review": "FAIL: missing citation", "attempts": 0}
        self.assertEqual(self.agent._route_after_verifier(failing_state), "fail")

        exhausted_state = {"review": "FAIL: missing citation", "attempts": 2}
        self.assertEqual(self.agent._route_after_verifier(exhausted_state), "pass")

    def test_run_stage_agent_uses_named_stage_agent(self) -> None:
        self.agent.stage_agents = {"supervisor": FakeStageAgent(["  hello  "])}

        result = self.agent._run_stage_agent("supervisor", "question", "trace")

        self.assertEqual(result, "hello")


if __name__ == "__main__":
    unittest.main()
