import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "05_code_review_agent.py"
SPEC = importlib.util.spec_from_file_location("code_review_agent", SCRIPT_PATH)
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)


class CodeReviewAgentTests(unittest.TestCase):
    def test_prompt_marks_source_as_untrusted(self):
        prompt = agent.build_prompt("print('hello')")
        self.assertIn("untrusted data", prompt)
        self.assertIn("print('hello')", prompt)

    def test_parse_valid_findings(self):
        response = json.dumps({"findings": [{
            "severity": "high",
            "title": "Unsafe operation",
            "line": 1,
            "explanation": "The operation can fail.",
            "recommendation": "Handle the failure.",
        }]})
        findings = agent.parse_findings(response, source_line_count=1)
        self.assertEqual(findings[0]["severity"], "high")
        self.assertEqual(findings[0]["line"], 1)

    def test_parse_rejects_invalid_line(self):
        response = json.dumps({"findings": [{
            "severity": "low",
            "title": "Issue",
            "line": 3,
            "explanation": "Reason.",
            "recommendation": "Fix.",
        }]})
        with self.assertRaises(ValueError):
            agent.parse_findings(response, source_line_count=2)

    def test_parse_rejects_invalid_json(self):
        with self.assertRaises(json.JSONDecodeError):
            agent.parse_findings("not json", source_line_count=1)

    def test_render_empty_findings_is_not_overconfident(self):
        text = agent.render_findings([])
        self.assertIn("not proof", text)

    def test_local_review_finds_eval_call(self):
        findings = agent.local_static_review("value = eval(user_input)")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line"], 1)
        self.assertIn("eval()", findings[0]["title"])

    def test_local_review_returns_syntax_error(self):
        findings = agent.local_static_review("def broken(:\n    pass")
        self.assertEqual(findings[0]["title"], "Python syntax error")
        self.assertEqual(findings[0]["line"], 1)

    def test_review_source_uses_api_and_returns_usage(self):
        usage = SimpleNamespace(input_tokens=24, output_tokens=12)
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"findings": []}')],
            usage=usage,
        )
        client = Mock()
        client.messages.create.return_value = response

        findings, returned_usage, latency = agent.review_source("print('ok')", client, "model-id")

        self.assertEqual(findings, [])
        self.assertIs(returned_usage, usage)
        self.assertGreaterEqual(latency, 0)
        client.messages.create.assert_called_once()
        self.assertEqual(client.messages.create.call_args.kwargs["model"], "model-id")


if __name__ == "__main__":
    unittest.main()
