"""
Day 12: Real LLM Integration - Provider Abstraction

Integrates concepts from Days 1-11 into a production-ready API layer:
- Structured outputs (Day 10)
- Error handling (Day 5, Day 10)
- Cost tracking (Day 1-3, Day 11)
- Context management (Day 2, Day 11)
- Model selection (Day 4)

Demonstrates:
- Multi-provider abstraction (Anthropic, OpenAI, Ollama mock)
- Retry logic with exponential backoff
- Streaming response handling
- Structured output enforcement
- Real cost tracking with provider pricing

Run: python experiments/12_api_providers.py
"""

import json
import os
import time
import httpx
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class TokenUsage:
    """Track token usage for cost estimation."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """Standardized response from any provider."""
    content: str
    usage: TokenUsage
    cost_usd: float
    provider: str
    model: str
    latency_seconds: float


# --- Provider Pricing (as of 2025) ---
# https://www.anthropic.com/pricing
# https://openai.com/api/pricing/
PROVIDER_PRICING = {
    "anthropic": {
        "claude-sonnet-4-20250514": {"input_per_1k": 0.003, "output_per_1k": 0.015},
        "claude-3-7-sonnet-20250219": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    },
    "openai": {
        "gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
        "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    },
    "ollama": {
        "qwen3.5:4b": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    },
}


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMProvider:
    """Abstract base for LLM API providers."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.pricing = self._get_pricing()

    def _get_pricing(self) -> Dict[str, float]:
        """Get pricing info for this provider + model."""
        provider_name = self.__class__.__name__.replace("Provider", "").lower()
        models = PROVIDER_PRICING.get(provider_name, {})
        return models.get(self.model, {"input_per_1k": 0.0, "output_per_1k": 0.0})

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost in USD for token usage."""
        input_cost = (usage.input_tokens / 1000) * self.pricing["input_per_1k"]
        output_cost = (usage.output_tokens / 1000) * self.pricing["output_per_1k"]
        return round(input_cost + output_cost, 6)

    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 1000,
             response_format: Optional[Dict] = None,
             stream: bool = False) -> LLMResponse:
        """Override in subclass."""
        raise NotImplementedError

    def stream_chat(self,
                    messages: List[Dict[str, str]],
                    temperature: float = 0.7,
                    max_tokens: int = 1000,
                    response_format: Optional[Dict] = None) -> Iterator[str]:
        """Override in subclass for streaming."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str = "claude-3-7-sonnet-20250219"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model, api_key)
        if not api_key:
            raise ProviderError("Set ANTHROPIC_API_KEY environment variable")

    def chat(self, messages, temperature=0.7, max_tokens=1000,
             response_format=None, stream=False) -> LLMResponse:
        if stream:
            # Non-streaming fallback
            chunks = list(self.stream_chat(messages, temperature, max_tokens, response_format))
            content = "".join(chunks)
            return LLMResponse(
                content=content,
                usage=TokenUsage(input_tokens=500, output_tokens=150),
                cost_usd=0.0,
                provider="anthropic",
                model=self.model,
                latency_seconds=0.5,
            )

        prompt = "\n".join(msg.get("content", "") for msg in messages)
        started_at = time.perf_counter()

        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                    **(
                        {"response_format": {"type": "json_object"}}
                        if response_format
                        else {}
                    ),
                },
                timeout=30.0,
            )
        except httpx.TimeoutException:
            raise ProviderError("Request timed out", retryable=True)
        except httpx.HTTPError as e:
            raise ProviderError(f"HTTP error: {e}", retryable=True)

        latency = time.perf_counter() - started_at

        if response.status_code == 401:
            raise ProviderError("Invalid API key", retryable=False)
        elif response.status_code == 429:
            raise ProviderError("Rate limited", retryable=True)
        elif response.status_code >= 500:
            raise ProviderError(f"Server error {response.status_code}", retryable=True)
        elif response.status_code != 200:
            raise ProviderError(f"Error {response.status_code}: {response.text[:200]}", retryable=False)

        data = response.json()
        content = data["content"][0]["text"] if data.get("content") else ""
        usage_data = data.get("usage", {})

        usage = TokenUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        )

        return LLMResponse(
            content=content,
            usage=usage,
            cost_usd=self.calculate_cost(usage),
            provider="anthropic",
            model=self.model,
            latency_seconds=round(latency, 3),
        )

    def stream_chat(self, messages, temperature=0.7, max_tokens=1000, response_format=None):
        prompt = "\n".join(msg.get("content", "") for msg in messages)
        started_at = time.perf_counter()

        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                    "stream": True,
                    **({"response_format": {"type": "json_object"}} if response_format else {}),
                },
                timeout=30.0,
            )
        except httpx.TimeoutException:
            raise ProviderError("Request timed out", retryable=True)

        latency = time.perf_counter() - started_at
        usage = TokenUsage(input_tokens=500, output_tokens=150)
        cost = self.calculate_cost(usage)

        yield f"[STREAM] Provider: {self.provider}, Model: {self.model}, Latency: {round(latency, 3)}s, Cost: ${cost}\n"
        yield f"[STREAM] Simulating: '{prompt[:50]}...'\n"
        yield f"[STREAM] Complete. Usage: {usage.total_tokens} tokens\n"


class OpenAIProvider(LLMProvider):
    """OpenAI GPT API provider."""

    def __init__(self, model: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(model, api_key)
        if not api_key:
            raise ProviderError("Set OPENAI_API_KEY environment variable")

    def chat(self, messages, temperature=0.7, max_tokens=1000,
             response_format=None, stream=False) -> LLMResponse:
        # Simulated call (requires real API key for actual use)
        prompt = "\n".join(msg.get("content", "") for msg in messages)
        started_at = time.perf_counter()

        usage = TokenUsage(
            input_tokens=max(10, len(prompt) // 4),
            output_tokens=100,
        )

        content = f"[OpenAI Provider - Simulated Response]\n"
        content += f"This is a simulated response from {self.model}.\n"
        content += f"Request: {prompt[:60]}...\n"

        latency = time.perf_counter() - started_at

        return LLMResponse(
            content=content,
            usage=usage,
            cost_usd=self.calculate_cost(usage),
            provider="openai",
            model=self.model,
            latency_seconds=round(latency, 3),
        )


class OllamaProvider(LLMProvider):
    """Local Ollama provider (mock for Day 12)."""

    def __init__(self, model: str = "qwen3.5:4b"):
        super().__init__(model, api_key="local")

    def chat(self, messages, temperature=0.7, max_tokens=1000,
             response_format=None, stream=False) -> LLMResponse:
        prompt = "\n".join(msg.get("content", "") for msg in messages)
        started_at = time.perf_counter()

        usage = TokenUsage(
            input_tokens=850,
            output_tokens=45,
        )

        content = f"[Ollama Local - Mock Response]\n"
        content += f"Model: {self.model}\n"
        content += f"Free local inference - no cost!\n"
        content += f"Note: Real Ollama calls require 'pip install ollipy' or direct HTTP to localhost:11434"

        latency = time.perf_counter() - started_at

        return LLMResponse(
            content=content,
            usage=usage,
            cost_usd=0.0,
            provider="ollama",
            model=self.model,
            latency_seconds=round(latency, 3),
        )

    def stream_chat(self, messages, temperature=0.7, max_tokens=1000, response_format=None):
        """Simulate streaming response."""
        prompt = "\n".join(msg.get("content", "") for msg in messages)

        # Simulate streaming by yielding chunks
        response_text = f"[Ollama Local - Streaming Response]\n"
        response_text += f"Model: {self.model}\n"
        response_text += f"Prompt: {prompt[:40]}...\n"
        response_text += f"Simulated streaming - this simulates a real Ollama response."

        for i in range(0, len(response_text), 10):
            yield response_text[i:i+10]
            time.sleep(0.01)

        yield f"\n[STREAM] Done. Cost: $0.00 (local)"


# --- Retry Logic ---

@contextmanager
def retry_provider(max_retries: int = 3, base_delay: float = 1.0):
    """Context manager for exponential backoff retries."""
    delays = []
    for attempt in range(max_retries):
        delays.append(base_delay * (2 ** attempt))
    yield delays


def call_with_retry(provider: LLMProvider,
                    messages: List[Dict[str, str]],
                    temperature: float = 0.7,
                    max_tokens: int = 1000,
                    response_format: Optional[Dict] = None,
                    max_retries: int = 3) -> LLMResponse:
    """Call LLM with exponential backoff retry logic."""

    delays = [1.0 * (2 ** i) for i in range(max_retries)]

    for attempt in range(max_retries):
        try:
            return provider.chat(messages, temperature, max_tokens, response_format)
        except ProviderError as e:
            if not e.retryable or attempt == max_retries - 1:
                raise
            print(f"  [RETRY {attempt + 1}/{max_retries}] Waiting {delays[attempt]}s...")
            time.sleep(delays[attempt])

    raise ProviderError("Max retries exceeded")


# --- Structured Output Enforcement ---

def enforce_structured_response(content: str, schema: Dict) -> Dict[str, Any]:
    """Parse and validate JSON response against schema."""
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try extracting JSON from response
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            result = json.loads(content[start:end + 1])
        else:
            raise ProviderError("No valid JSON in response")

    # Basic schema validation (simplified)
    required_keys = schema.get("required", [])
    for key in required_keys:
        if key not in result:
            raise ProviderError(f"Missing required key: {key}")

    return result


# --- Provider Manager ---

class ProviderManager:
    """Manage multiple LLM providers with fallback."""

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider = None

    def add_provider(self, name: str, provider: LLMProvider):
        """Register a provider."""
        self.providers[name] = provider
        if self.default_provider is None:
            self.default_provider = name

    def get_provider(self, name: str = None) -> LLMProvider:
        """Get a provider by name, or the default."""
        name = name or self.default_provider
        if name not in self.providers:
            raise ProviderError(f"Provider '{name}' not registered")
        return self.providers[name]

    def call(self,
             messages: List[Dict[str, str]],
             provider_name: str = None,
             temperature: float = 0.7,
             max_tokens: int = 1000,
             response_format: Optional[Dict] = None,
             max_retries: int = 3) -> LLMResponse:
        """Call LLM with provider fallback support."""

        # Try primary provider
        primary = provider_name or self.default_provider

        # Build provider list: primary first, then fallbacks
        provider_names = [primary] if primary else []
        for name in self.providers:
            if name not in provider_names:
                provider_names.append(name)

        last_error = None
        for pname in provider_names:
            try:
                provider = self.providers[pname]
                print(f"  [TRYING] Provider: {pname}")
                return call_with_retry(
                    provider, messages, temperature, max_tokens,
                    response_format, max_retries
                )
            except ProviderError as e:
                last_error = e
                print(f"  [FAILED] {pname}: {e}")
                continue

        raise last_error


# --- Demo Functions ---

DEMO_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["summary"],
}


def demo_basic_call(manager: ProviderManager) -> LLMResponse:
    """Demo 1: Basic API call across providers."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic API Call Across Providers")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "Explain what a context window is in 3 sentences."},
    ]

    response = manager.call(messages)
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_seconds}s")
    print(f"Tokens: {response.usage.input_tokens} input + {response.usage.output_tokens} output = {response.usage.total_tokens} total")
    print(f"Cost: ${response.cost_usd}")
    print(f"Response:\n{response.content}")

    return response


def demo_structured_output(manager: ProviderManager) -> Dict[str, Any]:
    """Demo 2: Structured output with schema validation."""
    print("\n" + "=" * 60)
    print("Demo 2: Structured Output with JSON Schema")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a technical documentation writer. "
                                      "Respond only in JSON matching the required schema."},
        {"role": "user", "content": "Summarize the concept of prompt caching in AI language models."},
    ]

    response = manager.call(messages, response_format=DEMO_SCHEMA)
    print(f"Raw response:\n{response.content}")

    try:
        structured = enforce_structured_response(response.content, DEMO_SCHEMA)
        print(f"\nParsed JSON:")
        print(json.dumps(structured, indent=2))
    except ProviderError as e:
        print(f"\nSchema validation failed: {e}")
        structured = {}

    return structured


def demo_streaming(provider: LLMProvider) -> None:
    """Demo 3: Streaming response handling."""
    print("\n" + "=" * 60)
    print("Demo 3: Streaming Response")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "Count from 1 to 10"},
    ]

    print("Streaming chunks:")
    for chunk in provider.stream_chat(messages):
        print(f"  {chunk}", end="", flush=True)
    print()


def demo_provider_comparison(manager: ProviderManager) -> None:
    """Demo 4: Compare costs across providers."""
    print("\n" + "=" * 60)
    print("Demo 4: Provider Cost Comparison")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "Explain transformers in 50 words."},
    ]

    print(f"{'Provider':<12} {'Model':<35} {'Input':>8} {'Output':>8} {'Total':>8} {'Cost($)':>10}")
    print("-" * 85)

    for name in manager.providers:
        provider = manager.providers[name]
        try:
            response = call_with_retry(provider, messages, max_retries=1)
            print(f"{response.provider:<12} {response.model:<35} "
                  f"{response.usage.input_tokens:>8} {response.usage.output_tokens:>8} "
                  f"{response.usage.total_tokens:>8} {response.cost_usd:>10.6f}")
        except ProviderError as e:
            print(f"{name:<12} {'ERROR':<35} {str(e):>8}")


def demo_retry_logic() -> None:
    """Demo 5: Retry with exponential backoff."""
    print("\n" + "=" * 60)
    print("Demo 5: Retry Logic (Exponential Backoff)")
    print("=" * 60)

    print("Delay schedule:")
    for attempt in range(3):
        delay = 1.0 * (2 ** attempt)
        print(f"  Attempt {attempt + 1}: wait {delay:.1f}s before retry")

    print("\nSimulating 3 retries with exponential backoff...")
    print("  [RETRY 1/3] Waiting 1.0s...")
    print("  [RETRY 2/3] Waiting 2.0s...")
    print("  [RETRY 3/3] Waiting 4.0s...")
    print("  [FAILED] Max retries exceeded")


# --- Main ---

def main():
    print("""
======================================================================
DAY 12: Real LLM Integration - Provider Abstraction
======================================================================

Integrates Days 1-11 concepts into a production-ready API layer:
- Structured outputs (Day 10)
- Error handling (Day 5, Day 10)
- Cost tracking (Days 1-3, Day 11)
- Context management (Days 2, Day 11)
- Model selection (Day 4)

Providers:
- Anthropic: Requires ANTHROPIC_API_KEY, real API calls
- OpenAI: Requires OPENAI_API_KEY, simulated response
- Ollama: Local model, mocked response (free)
""")

    # Initialize provider manager
    manager = ProviderManager()

    # Add simulated providers (works without API keys)
    manager.add_provider("ollama", OllamaProvider("qwen3.5:4b"))

    # Attempt to add real providers
    if os.getenv("OPENAI_API_KEY"):
        manager.add_provider("openai", OpenAIProvider("gpt-4o"))

    try:
        manager.add_provider("anthropic", AnthropicProvider())
    except ProviderError:
        print("  [INFO] ANTHROPIC_API_KEY not set - using simulated providers only")
        # Add a simulated Anthropic provider
        manager.add_provider("anthropic", OllamaProvider("claude-3-7-sonnet (simulated)"))

    # Run demos
    response = demo_basic_call(manager)
    structured = demo_structured_output(manager)
    demo_streaming(manager.get_provider("ollama"))
    demo_provider_comparison(manager)
    demo_retry_logic()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if structured:
        print(json.dumps(structured, indent=2))

    print(f"\nTotal cost across all demos: Tracked per-call")
    print(f"Providers tested: {list(manager.providers.keys())}")
    print(f"Default provider: {manager.default_provider}")

    print("""
KEY TAKEAWAYS:
1. Provider abstraction lets you switch between Anthropic/OpenAI/Ollama
2. Retry logic with exponential backoff handles transient failures
3. Structured outputs ensure reliable JSON parsing
4. Cost tracking enables informed provider selection
5. Fallback chains provide resilience when providers fail
6. Streaming improves user experience for long responses
""")


if __name__ == "__main__":
    main()
