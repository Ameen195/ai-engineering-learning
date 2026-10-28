# Day 10: LLM APIs - Structured Outputs and Error Handling

## API Call Structure

Every LLM API follows a similar pattern:

```python
# 1. Set up authentication
client = OpenAI(api_key="your-api-key")  # or Anthropic(), etc.

# 2. Format messages
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What is a token?"}
]

# 3. Make the call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.7,
    max_tokens=1000
)

# 4. Extract response
result = response.choices[0].message.content
```

## Key Parameters

| Parameter | Range | Use Case |
|-----------|-------|----------|
| **temperature** | 0.0-1.0 | 0.2=code/facts, 0.7=balanced, 1.0=creative |
| **max_tokens** | 1-Infinity | Controls response length |
| **top_p** | 0.0-1.0 | Nucleus sampling (usually 0.9-1.0) |
| **frequency_penalty** | 0.0-2.0 | Discourage repetition |
| **presence_penalty** | 0.0-2.0 | Encourage new topics |

## Structured Outputs

**Without schema:** LLM returns free text like "I think the score is 7 out of 10"
**With schema:** LLM returns strict JSON like `{"score": 7}`

### Why schemas matter:
- No parsing errors (you don't have to extract numbers from text)
- Predictable field names and types
- Can validate before processing
- Eliminates hallucinated formatting

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "code_review",
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 10},
                "issues": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["score", "issues"]
        }
    }
}
```

## Error Handling

APIs fail. Plan for it:

| HTTP Code | Cause | Strategy |
|-----------|-------|----------|
| **401** | Invalid API key | Check credentials |
| **429** | Rate limited | Exponential backoff (wait 1s, 2s, 4s...) |
| **400** | Bad request | Fix prompt or parameters |
| **500** | Server error | Retry after delay |

```python
def robust_call(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(...)
        except RateLimitError:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
        except APIError:
            time.sleep(1)
    return None
```

## Rate Limiting

Most providers enforce rate limits:
- **OpenAI**: 10,000 tokens/minute for free tier
- **Anthropic**: 4 requests/second (can be increased)
- **Groq**: Varies by model

**Best practices:**
1. Implement client-side rate limiting (token bucket)
2. Use exponential backoff for 429 errors
3. Cache responses to reduce API calls
4. Batch multiple requests when possible

## Cost Management

API calls aren't free. Estimate costs:

```python
# Input tokens: $0.003/1K tokens (GPT-4o)
# Output tokens: $0.006/1K tokens (GPT-4o)
input_cost = (input_tokens / 1000) * 0.003
output_cost = (output_tokens / 1000) * 0.006
total = input_cost + output_cost
```

**Cost reduction strategies:**
1. Use smaller models (GPT-3.5-turbo is 10x cheaper than GPT-4)
2. Cache frequent responses
3. Batch requests
4. Truncate unnecessary context
5. Use local models for development

## Local vs. Cloud APIs

| Factor | Local (Ollama) | Cloud (OpenAI) |
|--------|----------------|-----------------|
| Cost | Free | Pay per token |
| Quality | Good for 4-8B models | State-of-the-art |
| Speed | CPU-bound (slow) | Fast (GPU-optimized) |
| Privacy | Private | Data sent to provider |
| Setup | Install + download model | API key only |
| Reliability | Your hardware | Provider SLA |

## Files Created

- `experiments/10_api_patterns.py` - API patterns demonstration
- `notes/day10_apis.md` - This file

Run: `python experiments/10_api_patterns.py`
