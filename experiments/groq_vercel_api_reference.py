"""Reference-only Groq Vercel handler from the earlier deployment stage.

This module is retained for learning and tests; it is no longer deployed by Vercel.
"""

import hmac
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from http.server import BaseHTTPRequestHandler

from experiments.review_core import MAX_SOURCE_BYTES, build_prompt, parse_findings

MAX_BODY_BYTES = MAX_SOURCE_BYTES + 2_000
DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def json_response(status: int, payload: dict) -> tuple[int, dict]:
    return status, payload


def process_review(body: bytes, authorization: str, env: dict, opener=urlopen) -> tuple[int, dict]:
    if len(body) > MAX_BODY_BYTES:
        return json_response(413, {"error": "Request too large"})

    expected_password = env.get("APP_PASSWORD", "")
    provided_password = authorization.removeprefix("Bearer ")
    if len(expected_password) < 20 or not hmac.compare_digest(provided_password, expected_password):
        return json_response(401, {"error": "Unauthorized"})

    if not env.get("GROQ_API_KEY"):
        return json_response(503, {"error": "Review service is not configured"})

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return json_response(400, {"error": "Invalid JSON request"})

    source = payload.get("source") if isinstance(payload, dict) else None
    if not isinstance(source, str) or not source.strip():
        return json_response(400, {"error": "Provide Python source text"})
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return json_response(413, {"error": f"Source exceeds {MAX_SOURCE_BYTES:,} bytes"})

    model = env.get("GROQ_MODEL", DEFAULT_MODEL)
    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(source)}],
        "max_completion_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        GROQ_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {env['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with opener(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return json_response(502, {"error": f"Groq request failed (HTTP {error.code})"})
    except (URLError, TimeoutError, OSError):
        return json_response(502, {"error": "Could not reach Groq"})
    except (UnicodeError, json.JSONDecodeError):
        return json_response(502, {"error": "Groq returned an invalid response"})

    elapsed = time.perf_counter() - started_at
    try:
        text = result["choices"][0]["message"]["content"]
        findings = parse_findings(text, len(source.splitlines()))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return json_response(502, {"error": "Groq response did not contain valid review findings"})

    usage = result.get("usage") or {}
    return json_response(200, {
        "model": model,
        "findings": findings,
        "usage": {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in usage
        },
        "latency_seconds": round(elapsed, 2),
    })


class GroqReviewHandlerReference(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/review":
            self._send(404, {"error": "Not found"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._send(415, {"error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            self._send(400, {"error": "Invalid Content-Length"})
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._send(413 if length > MAX_BODY_BYTES else 400, {"error": "Invalid or oversized request"})
            return

        body = self.rfile.read(length)
        status, response = process_review(
            body,
            self.headers.get("Authorization", ""),
            os.environ,
        )
        self._send(status, response)

    def do_GET(self):
        self._send(405, {"error": "Method not allowed"})

    def _send(self, status: int, payload: dict):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
