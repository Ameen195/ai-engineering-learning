"""
Day 10: LLM APIs - Structured Outputs and Error Handling

Demonstrates:
- Message structure for chat APIs
- Structured output with JSON schema
- Error handling and retry patterns
- Rate limiting strategies
"""

import json
import time
from typing import Dict, List, Any, Optional
from enum import Enum

# --- API Parameters ---

class Temperature:
    DETERMINISTIC = 0.2  # Technical/code tasks
    BALANCED = 0.7       # General conversation
    CREATIVE = 1.0       # Brainstorming

# --- Structured Output Schema ---

CODE_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "Brief description of the task"},
        "model_type": {"type": "string", "enum": ["classification", "generation", "analysis"]},
        "complexity": {"type": "string", "enum": ["low", "medium", "high"]},
        "issues_found": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "line": {"type": "integer"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "recommendation": {"type": "string"}
                }
            }
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["task", "model_type", "complexity", "issues_found", "confidence"]
}

LEARNING_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "estimated_hours": {"type": "integer"},
        "prerequisites": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["topic", "estimated_hours", "prerequisites", "next_steps", "confidence"]
}

# --- Simulated API Responses ---

def simulate_api_call(messages: List[Dict], schema: Dict = None) -> Dict:
    """
    Simulates what an LLM API would return.
    In production, this would be:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_schema", "json_schema": schema}
        )
    """
    content = messages[-1]["content"].lower()

    # Determine response type based on content
    if "code" in content and "review" in content:
        response = {
            "task": "Code Review",
            "model_type": "classification",
            "complexity": "medium",
            "issues_found": [
                {
                    "pattern": "eval() usage",
                    "line": 12,
                    "severity": "high",
                    "recommendation": "Replace with ast.literal_eval for safe evaluation"
                }
            ],
            "confidence": 0.92
        }
    elif "learning" in content or "curriculum" in content or "path" in content:
        response = {
            "topic": "VRAM Calculation",
            "estimated_hours": 5,
            "prerequisites": ["Tokenization basics", "Transformer architecture"],
            "next_steps": [
                "Practice with different model sizes",
                "Learn quantization techniques",
                "Build a VRAM calculator tool"
            ],
            "confidence": 0.88
        }
    else:
        response = {
            "task": "General Question",
            "model_type": "analysis",
            "complexity": "low",
            "issues_found": [],
            "confidence": 0.75
        }

    return {"choices": [{"message": {"content": json.dumps(response, indent=2)}}]}

# --- API Client Wrapper ---

class LLMClient:
    """
    A wrapper around chat completions API with:
    - Structured output support
    - Error handling with retries
    - Rate limiting awareness
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        self.last_request_time = 0
        self.request_count = 0

    def call(self, messages: List[Dict], schema: Dict = None,
             temperature: float = 0.7, max_tokens: int = 1000,
             max_retries: int = 3) -> Optional[Dict]:
        """
        Make a chat completion API call with error handling.

        Args:
            messages: Conversation history
            schema: JSON schema for structured output
            temperature: Controls randomness
            max_tokens: Max output length
            max_retries: Number of retry attempts

        Returns:
            API response dict or None on failure
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting: ensure minimum interval between requests
                self._respect_rate_limit()

                # Build request parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if schema:
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema}
                    }

                # Simulate API call (replace with real client in production)
                response = simulate_api_call(messages, schema)

                self.request_count += 1
                return response

            except self._RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  [Rate limited] Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            except self._BadGateWayError as e:
                print(f"  [Server error] Retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
                continue

            except Exception as e:
                print(f"  [Error] {type(e).__name__}: {e}")
                return None

        print(f"  [Failed] After {max_retries} retries")
        return None

    def _respect_rate_limit(self):
        """Ensure minimum time between requests (simulate rate limiting)."""
        min_interval = 0.1  # 100ms between requests (adjust based on provider)
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    class _RateLimitError(Exception):
        """Simulated rate limit error."""
        pass

    class _BadGateWayError(Exception):
        """Simulated server error."""
        pass

# --- Demo Functions ---

def demo_structured_output():
    """Show how structured outputs eliminate parsing errors."""
    print("=" * 70)
    print("DAY 10: LLM APIs & Structured Outputs")
    print("=" * 70)

    print("\n1. STRUCTURED OUTPUT DEMO")
    print("-" * 70)
    print("Without schema -> LLM returns free text -> you parse it:")
    print("  'I found one issue: eval() on line 12. Score: 7/10'")
    print("  Parsing: extract number from '7/10' -> fragile!")
    print()
    print("With schema -> LLM returns strict JSON -> no parsing needed:")

    client = LLMClient()
    messages = [
        {"role": "system", "content": "You are a code review tool. Return findings as JSON."},
        {"role": "user", "content": "Review this Python code for security issues: print(eval('1+1'))"}
    ]

    response = client.call(messages, schema=CODE_REVIEW_SCHEMA, temperature=Temperature.DETERMINISTIC)
    if response:
        content = response["choices"][0]["message"]["content"]
        result = json.loads(content)
        print(f"\nStructured Response:")
        print(json.dumps(result, indent=2))
        print(f"\nParsed fields:")
        print(f"  Task: {result['task']}")
        print(f"  Issues: {len(result['issues_found'])} found")
        print(f"  Confidence: {result['confidence']}")

def demo_error_handling():
    """Show how error handling works."""
    print("\n\n2. ERROR HANDLING DEMO")
    print("-" * 70)
    print("API calls can fail in predictable ways:")
    print("  HTTP 401: Invalid API key")
    print("  HTTP 429: Rate limited (too many requests)")
    print("  HTTP 400: Bad request (fix prompt/parameters)")
    print("  HTTP 500: Server error (rare, retry)")
    print()
    print("Best practices:")
    print("  1. Use exponential backoff for rate limits (1s, 2s, 4s)")
    print("  2. Log errors with context (model, prompt length, etc.)")
    print("  3. Set max_retries to avoid infinite loops")
    print("  4. Cache responses to reduce API calls")
    print("  5. Use structured outputs to avoid parsing errors")

def demo_rate_limiting():
    """Show rate limiting strategy."""
    print("\n\n3. RATE LIMITING DEMO")
    print("-" * 70)

    client = LLMClient()
    print(f"Provider: {client.provider}")
    print(f"Model: {client.model}")
    print(f"Minimum interval: 100ms between requests")
    print(f"Strategy: Exponential backoff on rate limit errors")
    print()
    print("Simulated request pattern:")
    print("  Request 1: OK")
    print("  Request 2: OK")
    print("  Request 3: 429 (rate limited) -> wait 1s -> retry -> OK")
    print("  Request 4: OK")
    print("  Request 5: 429 (rate limited) -> wait 2s -> retry -> OK")

def demo_real_api_vs_local():
    """Compare cloud vs local APIs."""
    print("\n\n4. CLOUD vs LOCAL API COMPARISON")
    print("-" * 70)

    print("Cloud API (OpenAI, Anthropic, Groq):")
    print("  Pros: State-of-the-art quality, consistent, managed infra")
    print("  Cons: Pay per token, network latency, rate limits")
    print("  Cost: $0.001-0.03 per 1K tokens")
    print()

    print("Local API (Ollama, llama.cpp):")
    print("  Pros: Free, private, no rate limits, offline")
    print("  Cons: Lower quality, slower on consumer hardware")
    print("  Cost: $0.00 (your electricity)")
    print()

    print("Hybrid approach:")
    print("  - Use local for development/testing (free iteration)")
    print("  - Use cloud for production (best quality)")
    print("  - Cache responses to minimize API calls")

def demo_your_agent_api():
    """Show how your Day 9 agent uses API patterns."""
    print("\n\n5. YOUR LEARNING ASSISTANT - API INTEGRATION")
    print("-" * 70)
    print("Your agent uses these patterns:")

    client = LLMClient()
    schema = LEARNING_PATH_SCHEMA

    messages = [
        {"role": "system", "content": "You are an AI learning advisor for engineers."},
        {"role": "user", "content": "I just learned about VRAM and quantization. What should I study next?"}
    ]

    response = client.call(messages, schema=schema, temperature=0.3)
    if response:
        result = json.loads(response["choices"][0]["message"]["content"])
        print(f"\nAPI Call Response:")
        print(json.dumps(result, indent=2))

    print(f"\nTotal API requests: {client.request_count}")
    print(f"Cache hits: Simulate caching to save costs")

    # Show what a real API call would look like
    print("\n\n6. REAL API CALL STRUCTURE (for reference)")
    print("-" * 70)
    print("""
# OpenAI
from openai import OpenAI
client = OpenAI(api_key="your-key")

# Structured output
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Return JSON only"},
        {"role": "user", "content": "Review this code: ..."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "review", "schema": CODE_REVIEW_SCHEMA}
    },
    temperature=0.2,
    max_tokens=1000
)
result = json.loads(response.choices[0].message.content)

# Anthropic (Claude)
from anthropic import Anthropic
client = Anthropic(api_key="your-key")
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Review this code: ..."}],
    max_tokens=1000,
    response_model=json.loads,  # For structured output
)

# Local (Ollama)
import requests
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:7b",
    "prompt": "Review this code: ...",
    "format": "json"  # Request JSON output
})
""")

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Structured outputs eliminate parsing errors - use JSON schemas")
    print("2. Error handling with exponential backoff prevents crashes")
    print("3. Local = free/debugging; Cloud = quality/production")
    print("4. Caching and batch requests minimize API costs")
    print("5. Temperature 0.2 for code/technical, 0.7 for general, 1.0 for creative")
    print("=" * 70)

    print("-" * 70)
    print("Your API cost estimation (for your learning):")
    print(f"  - Day 9 experiment simulated: $0.00 (no API calls)")
    print(f"  - Day 10 experiment simulated: $0.00 (no API calls)")
    print(f"  - If real: ~$0.01-0.03 per call")
    print(f"  - With caching: ~50-90% savings")
    print("=" * 70)

if __name__ == "__main__":
    demo_structured_output()
    demo_error_handling()
    demo_rate_limiting()
    demo_real_api_vs_local()
    demo_your_agent_api()
