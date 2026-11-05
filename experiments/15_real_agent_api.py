"""Day 15: a real provider-backed learning assistant.

This teaching example makes provider calls only; it never silently substitutes
mock text for a model response. Run with AI_LAB_PROVIDER=ollama (default), or
explicitly select openai, anthropic, or auto through the environment.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None
    HAS_ANTHROPIC = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False


HARDWARE_PROFILE = {
    "gpu": "NVIDIA GTX 1650",
    "vram_gb": 4.0,
    "ram_gb": 16,
    "cpu": "Intel Core i7-10870H",
    "os": "Windows 64-bit",
}

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "tools_used": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "estimated_hours": {"type": "number"},
                },
                "required": ["title", "description", "priority", "estimated_hours"],
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["query_type", "confidence", "tools_used", "answer", "recommendations", "limitations", "sources"],
}


class ProviderError(Exception):
    """A provider failure with safe diagnostic metadata for callers."""

    def __init__(self, message: str, retryable: bool = False, *, code: str = "provider_error",
                 provider: Optional[str] = None, model: Optional[str] = None):
        super().__init__(message)
        self.retryable = retryable
        self.code = code
        self.provider = provider
        self.model = model


def _http_error(status: int, detail: str) -> ProviderError:
    """Translate HTTP status into a conservative retry decision."""
    retryable = status in {408, 409, 429, 500, 502, 503, 504}
    return ProviderError(f"HTTP {status}: {detail[:240]}", retryable=retryable, code=f"http_{status}")


class LLMProvider:
    name = "base"
    model_name = "unknown"
    input_price_env = ""
    output_price_env = ""

    def is_available(self) -> bool:
        raise NotImplementedError

    def chat(self, messages: List[Dict[str, str]], temperature: float,
             max_tokens: int) -> Dict[str, Any]:
        raise NotImplementedError

    def estimate_cost(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate USD only when both user-supplied prices are configured.

        Prices are USD per million tokens. No provider price is hard-coded.
        """
        in_price = os.getenv(self.input_price_env, "") if self.input_price_env else ""
        out_price = os.getenv(self.output_price_env, "") if self.output_price_env else ""
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if in_price and out_price and input_tokens is not None and output_tokens is not None:
            try:
                amount = (input_tokens * float(in_price) + output_tokens * float(out_price)) / 1_000_000
                return {"usd": amount, "source": "estimate_from_user_configured_rates"}
            except ValueError:
                pass
        if self.name == "ollama":
            return {"usd": 0.0, "source": "local_inference_no_provider_api_fee"}
        return {"usd": None, "source": "unavailable_rate_or_usage_missing"}


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    input_price_env = "ANTHROPIC_INPUT_USD_PER_MILLION"
    output_price_env = "ANTHROPIC_OUTPUT_USD_PER_MILLION"

    def __init__(self):
        self.model_name = os.getenv("ANTHROPIC_MODEL", "")

    def is_available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY")) and bool(self.model_name) and HAS_ANTHROPIC

    def chat(self, messages, temperature=0.3, max_tokens=1000):
        if not HAS_ANTHROPIC:
            raise ProviderError("Install the anthropic package to use this provider.")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")

        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        request_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m.get("role") != "system"
        ]
        try:
            response = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=30.0).messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system="\n\n".join(system_parts),
                messages=request_messages,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status:
                raise _http_error(status, str(exc)) from exc
            raise ProviderError(f"Anthropic request failed: {exc}", retryable=True) from exc

        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None
        content = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return {
            "content": content,
            "usage": _usage(in_tokens, out_tokens, "provider_reported" if usage else "unavailable"),
            "provider": self.name,
            "model": self.model_name,
        }


class OpenAIProvider(LLMProvider):
    name = "openai"
    input_price_env = "OPENAI_INPUT_USD_PER_MILLION"
    output_price_env = "OPENAI_OUTPUT_USD_PER_MILLION"

    def __init__(self):
        self.model_name = os.getenv("OPENAI_MODEL", "")

    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY")) and bool(self.model_name) and HAS_OPENAI

    def chat(self, messages, temperature=0.3, max_tokens=1000):
        if not HAS_OPENAI:
            raise ProviderError("Install the openai package to use this provider.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set.")
        try:
            client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status:
                raise _http_error(status, str(exc)) from exc
            raise ProviderError(f"OpenAI request failed: {exc}", retryable=True) from exc

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""
        reported = getattr(response, "usage", None)
        in_tokens = getattr(reported, "prompt_tokens", None) if reported else None
        out_tokens = getattr(reported, "completion_tokens", None) if reported else None
        return {
            "content": content or "",
            "usage": _usage(in_tokens, out_tokens, "provider_reported" if reported else "unavailable"),
            "provider": self.name,
            "model": self.model_name,
        }


def _usage(input_tokens: Optional[int], output_tokens: Optional[int], source: str) -> Dict[str, Any]:
    """Keep missing usage missing; never turn it into zero or a guess."""
    if input_tokens is None or output_tokens is None:
        source = "partially_reported" if input_tokens is not None or output_tokens is not None else "unavailable"
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "source": source,
    }


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model_name = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

    def is_available(self) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                return False
            model_names = {m.get("name") for m in response.json().get("models", [])}
            return self.model_name in model_names
        except (requests.RequestException, ValueError, AttributeError):
            return False

    def chat(self, messages, temperature=0.3, max_tokens=1000):
        if not HAS_REQUESTS:
            raise ProviderError("Install requests to connect to Ollama.")
        think = os.getenv("OLLAMA_THINK", "false").strip().lower() in {"1", "true", "yes", "on"}
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    # Qwen's default thinking can spend a long time reasoning before
                    # producing the compact structured answer this app needs.
                    "think": think,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120,
            )
        except requests.Timeout as exc:
            raise ProviderError(
                "Ollama did not finish before the request timeout.", retryable=True,
                code="timeout", provider=self.name, model=self.model_name,
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(
                f"Ollama request failed: {exc}", retryable=True, code="connection_error",
                provider=self.name, model=self.model_name,
            ) from exc
        if response.status_code != 200:
            error = _http_error(response.status_code, response.text)
            error.provider, error.model = self.name, self.model_name
            raise error
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                "Ollama returned invalid JSON.", code="invalid_response",
                provider=self.name, model=self.model_name,
            ) from exc
        return {
            "content": data.get("message", {}).get("content", ""),
            "usage": _usage(data.get("prompt_eval_count"), data.get("eval_count"), "provider_reported"),
            "provider": self.name,
            "model": self.model_name,
        }


class FallbackProviderChain:
    """Try selected providers in order, retrying only transient failures."""

    def __init__(self, providers: List[LLMProvider], max_retries: int = 2,
                 retry_delays: Optional[List[float]] = None):
        self.providers = providers
        self.max_retries = max(1, max_retries)
        self.retry_delays = retry_delays or [1.0]

    def chat(self, messages, temperature=0.3, max_tokens=1000):
        errors = []
        last_error = None
        last_provider = None
        last_model = None
        for provider in self.providers:
            if not provider.is_available():
                errors.append(f"{provider.name}: unavailable (check key, SDK, server, and model)")
                last_provider, last_model = provider.name, provider.model_name
                last_error = "unavailable"
                continue
            for attempt in range(self.max_retries):
                try:
                    result = provider.chat(messages, temperature, max_tokens)
                    result["attempt_number"] = attempt + 1
                    result["cost"] = provider.estimate_cost(result["usage"])
                    return result
                except ProviderError as exc:
                    errors.append(f"{provider.name}: {exc}")
                    last_provider, last_model = exc.provider or provider.name, exc.model or provider.model_name
                    last_error = exc
                    if not exc.retryable or attempt + 1 == self.max_retries:
                        break
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    time.sleep(delay)
        raise ProviderError(
            "All configured providers failed: " + "; ".join(errors),
            retryable=bool(last_error and getattr(last_error, "retryable", False)),
            code=getattr(last_error, "code", last_error or "provider_error"),
            provider=last_provider, model=last_model,
        )


def _validate_recommendation(data: Any) -> List[str]:
    """Validate the response fields used by this app (small JSON-schema subset)."""
    problems = []
    if not isinstance(data, dict):
        return ["Response must be a JSON object."]
    required = RECOMMENDATION_SCHEMA["required"]
    for key in required:
        if key not in data:
            problems.append(f"Missing required field: {key}")
    expected = {
        "query_type": str, "confidence": (int, float), "tools_used": list,
        "answer": str, "recommendations": list, "limitations": list, "sources": list,
    }
    for key, kind in expected.items():
        if key in data and (isinstance(data[key], bool) or not isinstance(data[key], kind)):
            problems.append(f"{key} has the wrong type.")
    if isinstance(data.get("confidence"), (int, float)) and not 0 <= data["confidence"] <= 1:
        problems.append("confidence must be between 0 and 1.")
    for key in ("tools_used", "limitations", "sources"):
        if isinstance(data.get(key), list) and not all(isinstance(item, str) for item in data[key]):
            problems.append(f"Every item in {key} must be a string.")
    if isinstance(data.get("recommendations"), list):
        for i, item in enumerate(data["recommendations"]):
            if not isinstance(item, dict):
                problems.append(f"recommendations[{i}] must be an object.")
                continue
            for key in ("title", "description", "priority", "estimated_hours"):
                if key not in item:
                    problems.append(f"recommendations[{i}] is missing {key}.")
            if item.get("priority") not in {"high", "medium", "low"}:
                problems.append(f"recommendations[{i}].priority must be high, medium, or low.")
            hours = item.get("estimated_hours")
            if isinstance(hours, bool) or not isinstance(hours, (int, float)):
                problems.append(f"recommendations[{i}].estimated_hours must be a number.")
    return problems


class RealLearningAssistantAgent:
    def __init__(self, hardware_profile: Dict[str, Any], chain: Optional[FallbackProviderChain] = None):
        self.hardware = hardware_profile
        self.chain = chain or self._configured_chain()
        self.request_count = 0
        self.reported_input_tokens = 0
        self.reported_output_tokens = 0
        self.usage_unavailable_requests = 0
        self.estimated_cost_usd = 0.0
        self.estimated_cost_requests = 0

    @staticmethod
    def _configured_chain() -> FallbackProviderChain:
        providers_by_name = {
            "anthropic": AnthropicProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider,
        }
        selected = os.getenv("AI_LAB_PROVIDER", "ollama").strip().lower()
        if selected == "auto":
            order = ["anthropic", "openai", "ollama"]
            print("Provider mode: auto; a fallback provider may be used after failure.")
        elif selected in providers_by_name:
            order = [selected]
            print(f"Provider mode: {selected}; no paid provider fallback is enabled.")
        else:
            raise ValueError("AI_LAB_PROVIDER must be ollama, openai, anthropic, or auto.")
        return FallbackProviderChain([providers_by_name[name]() for name in order])

    def _system_prompt(self) -> str:
        return (
            "You are an AI Engineering Learning Assistant. Answer clearly and concisely. "
            "Return only a JSON object with fields query_type, confidence, tools_used, answer, "
            "recommendations, limitations, and sources. Each recommendation must contain title, "
            "description, priority (high/medium/low), and estimated_hours (integer). "
            "Do not claim tools were used unless they were actually executed.\n"
            f"Hardware profile: {json.dumps(self.hardware)}\n"
            f"Required response shape: {json.dumps(RECOMMENDATION_SCHEMA)}"
        )

    def query(self, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        self.request_count += 1
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": question},
        ]
        try:
            response = self.chain.chat(messages, temperature=0.3, max_tokens=1000)
        except ProviderError as exc:
            self.usage_unavailable_requests += 1
            return {
                "status": "provider_error", "answer": str(exc), "provider": exc.provider,
                "model": exc.model, "error_code": exc.code,
                "usage": _usage(None, None, "unavailable"),
                "cost": {"usd": None, "source": "unavailable"},
                "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
            }

        usage = response["usage"]
        cost = response["cost"]
        if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
            self.reported_input_tokens += usage["input_tokens"]
            self.reported_output_tokens += usage["output_tokens"]
        if usage["source"] != "provider_reported":
            self.usage_unavailable_requests += 1
        if cost["usd"] is not None and cost["source"] == "estimate_from_user_configured_rates":
            self.estimated_cost_usd += cost["usd"]
            self.estimated_cost_requests += 1

        try:
            result = json.loads(response["content"])
            problems = _validate_recommendation(result)
        except (json.JSONDecodeError, TypeError):
            result, problems = {}, ["Provider response was not valid JSON."]
        if problems:
            result = {
                "status": "invalid_response",
                "answer": response.get("content", ""),
                "validation_errors": problems,
                "limitations": ["The model output did not match the expected response structure."],
            }
        else:
            result["status"] = "ok"
        result.update({
            "provider": response["provider"],
            "model": response["model"],
            "usage": usage,
            "cost": cost,
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "requests": self.request_count,
            "provider_reported_input_tokens": self.reported_input_tokens,
            "provider_reported_output_tokens": self.reported_output_tokens,
            "requests_without_complete_usage": self.usage_unavailable_requests,
            "sum_of_configured_cost_estimates_usd": round(self.estimated_cost_usd, 8),
            "requests_with_configured_cost_estimate": self.estimated_cost_requests,
        }


def main() -> None:
    print("Day 15 — real provider-backed learning assistant")
    print("Set AI_LAB_PROVIDER=ollama (default), openai, anthropic, or auto.")
    print("Enter one question. Retries or auto fallback can send additional requests.")
    print("Type 'exit' to quit.\n")
    question = input("Question: ").strip()
    if not question or question.lower() == "exit":
        return
    agent = RealLearningAssistantAgent(HARDWARE_PROFILE)
    result = agent.query(question)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Session stats:")
    print(json.dumps(agent.get_stats(), indent=2))


if __name__ == "__main__":
    main()
