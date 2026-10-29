# Day 12: Real LLM Integration - Provider Abstraction

## What We Built

A production-ready LLM provider abstraction layer that supports multiple backends:
- Anthropic (Claude API)
- OpenAI (GPT API)
- Ollama (local models)

## Key Concepts

### 1. Provider Abstraction
```python
class LLMProvider:
    """Abstract interface for any LLM provider."""
    def chat(self, messages, temperature, max_tokens, response_format) -> LLMResponse
    def stream_chat(self, messages, ...) -> Iterator[str]
```

This lets you switch providers without changing client code.

### 2. Retry Logic with Exponential Backoff
```python
delays = [1.0, 2.0, 4.0, 8.0]  # 1s, 2s, 4s, 8s
for attempt in range(max_retries):
    try:
        return provider.chat(...)
    except ProviderError as e:
        if not e.retryable:
            raise
        time.sleep(delays[attempt])
```

**Why:** API calls fail due to network issues, rate limits (429), or server errors (500+). Exponential backoff prevents hammering the API during outages.

### 3. Structured Output Enforcement
```python
def enforce_structured_response(content: str, schema: Dict) -> Dict[str, Any]:
    """Parse and validate JSON response against schema."""
    result = json.loads(content)
    # Validate required fields
    for key in schema.get("required", []):
        if key not in result:
            raise ProviderError(f"Missing required key: {key}")
    return result
```

**Why:** LLMs sometimes return malformed JSON or omit fields. Schema validation ensures reliable parsing.

### 4. Provider Manager with Fallback
```python
manager = ProviderManager()
manager.add_provider("anthropic", AnthropicProvider())
manager.add_provider("openai", OpenAIProvider())
manager.add_provider("ollama", OllamaProvider())

result = manager.call(messages, provider_name="anthropic")
# If Anthropic fails, falls back to OpenAI, then Ollama
```

**Why:** Production systems need resilience. If one provider is down or rate-limited, automatically fall back to another.

### 5. Cost Tracking
```python
PROVIDER_PRICING = {
    "anthropic": {"claude-sonnet-4": {"input_per_1k": 0.003, "output_per_1k": 0.015}},
    "openai": {"gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01}},
    "ollama": {"qwen3.5:4b": {"input_per_1k": 0.0, "output_per_1k": 0.0}},
}
```

Tracks real costs per provider for informed decision-making.

## Design Decisions

1. **LLMResponse dataclass**: Standardizes responses regardless of provider
2. **ProviderError with retryable flag**: Distinguishes transient vs. permanent errors
3. **Streaming support**: For better UX with long responses
4. **Fallback chains**: Resilience against provider failures
5. **Pricing registry**: Central source for cost calculations

## Provider Comparison

| Provider | Setup Required | Cost | Speed | Privacy | Use Case |
|----------|---------------|------|-------|---------|----------|
| Anthropic | API key | $0.003/1K in | Fast | Third-party | General + long ctx |
| OpenAI | API key | $0.0025/1K in | Fast | Third-party | Fast + reliable |
| Ollama | Local install | Free | Slow | Local | Privacy + free |

## Key Takeaways

1. **Provider abstraction** enables flexible, swappable backends
2. **Retry logic** prevents failures from crashing your application
3. **Structured outputs** make LLM responses programmatically usable
4. **Fallback chains** ensure availability even when providers fail
5. **Cost tracking** enables informed provider selection
6. **Streaming** improves perceived performance for users

## Integration with Previous Days

| Component | From Day | How Used |
|-----------|----------|----------|
| Token counting | Day 1 | Estimate input/output tokens |
| Context management | Day 2 | Track conversation length |
| Prompt caching | Day 3 | Cache frequent queries |
| Model comparison | Day 4 | Select best provider |
| Error handling | Day 5, 10 | Graceful provider failures |
| Cost tracking | Day 1-3 | Per-provider pricing |

## Files Created

- `experiments/12_api_providers.py` - Provider abstraction layer
- `notes/day12_api_providers.md` - This file

## Run

```bash
python experiments/12_api_providers.py
```

## Next Steps

- Add real API calls (requires API keys)
- Add provider health checks
- Add automatic provider selection based on cost/latency
- Add prompt caching layer
