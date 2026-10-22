import importlib.util
import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

API_PATH = Path(__file__).parents[1] / "experiments" / "groq_vercel_api_reference.py"
SPEC = importlib.util.spec_from_file_location("vercel_review_api", API_PATH)
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)

ENV = {"GROQ_API_KEY": "test-secret", "APP_PASSWORD": "demo-password-at-least-20"}


def fake_opener(response_body):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_body).encode()

    def opener(request, timeout):
        return Response()

    return opener


class VercelReviewApiTests(unittest.TestCase):
    def test_rejects_unauthorized_request(self):
        status, body = api.process_review(b'{"source":"print(1)"}', "Bearer wrong", ENV)
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "Unauthorized")

    def test_rejects_invalid_json(self):
        status, _ = api.process_review(b"not json", "Bearer demo-password-at-least-20", ENV)
        self.assertEqual(status, 400)

    def test_rejects_short_app_password_configuration(self):
        status, body = api.process_review(
            b'{"source":"print(1)"}',
            "Bearer short",
            {"GROQ_API_KEY": "test-secret", "APP_PASSWORD": "short"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "Unauthorized")

    def test_rejects_oversized_body(self):
        body = b" " * (api.MAX_BODY_BYTES + 1)
        status, _ = api.process_review(body, "Bearer demo-password-at-least-20", ENV)
        self.assertEqual(status, 413)

    def test_returns_validated_groq_findings_and_usage(self):
        provider_response = {
            "choices": [{"message": {"content": json.dumps({"findings": [{
                "severity": "high",
                "title": "Unsafe eval",
                "line": 1,
                "explanation": "Dynamic input is evaluated.",
                "recommendation": "Use a safe parser.",
            }]})}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
        }
        status, body = api.process_review(
            json.dumps({"source": "eval(user_input)"}).encode(),
            "Bearer demo-password-at-least-20",
            ENV,
            opener=fake_opener(provider_response),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["findings"][0]["line"], 1)
        self.assertEqual(body["usage"]["prompt_tokens"], 40)
        self.assertEqual(body["model"], "openai/gpt-oss-20b")

    def test_does_not_expose_upstream_error_body(self):
        from urllib.error import HTTPError

        def failing_opener(request, timeout):
            raise HTTPError(request.full_url, 429, "quota", {}, BytesIO(b"secret details"))

        status, body = api.process_review(
            json.dumps({"source": "print(1)"}).encode(),
            "Bearer demo-password-at-least-20",
            ENV,
            opener=failing_opener,
        )
        self.assertEqual(status, 502)
        self.assertNotIn("secret details", json.dumps(body))

    def test_handler_rejects_non_json_content_type(self):
        class FakeHandler(api.GroqReviewHandlerReference):
            def __init__(self):
                self.path = "/api/review"
                self.headers = {"Content-Type": "text/plain"}
                self.responses = []

            def _send(self, status, payload):
                self.responses.append((status, payload))

        instance = FakeHandler()
        instance.do_POST()
        self.assertEqual(instance.responses[0][0], 415)


if __name__ == "__main__":
    unittest.main()
