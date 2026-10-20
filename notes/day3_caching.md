# Day 3: Prompt Caching and Optimization

## What I learned

### The problem: Context snowball
- Each conversation turn re-sends the entire history
- Turn 1: 100 tokens → Turn 10: 1000 tokens → Turn 20: 2000 tokens
- You pay full price for the same tokens over and over
- This makes long agent conversations very expensive

### The solution: Prompt caching
- If the beginning of your prompt (prefix) matches a previous request, the provider reuses the cached computation
- You get a discount on the cached portion (50-90% off)
- Only the new part (your latest question) is charged at full price

### How caching works
```
Request: [System (200)] + [History (500)] + [New question (50)]
                                    ↑ cached ↑         ↑ new ↑
                                    
Without cache: Pay for all 750 tokens at full price
With cache:    Pay for 700 tokens at discounted rate + 50 at full price
```

### Provider comparison (as of early 2025)

| Provider | Discount | Min cache size | TTL |
|----------|----------|----------------|-----|
| Anthropic | 90% | 1024 tokens | 5 minutes |
| OpenAI | 50% | 1024 tokens | 1 hour |
| Google | 75% | 32,768 tokens | 1 hour |

### What breaks the cache
- Changing the system message (even adding one word)
- Reordering conversation history
- Modifying any character in old messages
- Adding whitespace

**Rule:** Keep the prefix stable. Only add new messages at the END.

### Minimum cache size impact
- Most providers require at least 1024 tokens to cache
- Short conversations (< 1024 tokens) don't benefit
- For agents: Early turns won't cache, but later turns will

### Real-world savings example

10-turn coding agent conversation:
- Without caching: $0.015
- With Anthropic caching (90%): $0.003
- **Savings: 80%**

For a 50-turn agent with tool calls:
- Without caching: $0.50
- With caching: $0.095
- **Savings: 81%**

### Cache expiration
- Anthropic: 5 minutes of inactivity
- OpenAI: 1 hour of inactivity
- After expiration, the next request pays full price again
- Frequent turns keep the cache warm

### Key insights for AI engineering

1. **Caching is essential for production agents** - not optional
2. **Structure prompts for cache hits** - stable system message, append-only history
3. **Cache expires** - design for your use case (interactive vs batch)
4. **Minimum size matters** - short conversations don't benefit
5. **Not all providers support it** - check documentation
6. **Storage costs** - some providers charge to keep data in cache

## Vocabulary
- **Prompt caching**: Reusing computation for repeated prompt prefixes
- **Cache hit**: When the prefix matches a cached request
- **Cache miss**: When the prefix doesn't match (full price)
- **Prefix**: The stable beginning of the prompt (system message + history)
- **TTL (Time To Live)**: How long cached data persists before expiring
- **Cache storage cost**: Fee for keeping data in cache (per hour)

## Code patterns
```python
# Calculate cost with caching
cached_tokens = 700  # System + history
uncached_tokens = 50  # New question
output_tokens = 100

cached_cost = (cached_tokens / 1_000_000) * discounted_price
uncached_cost = (uncached_tokens / 1_000_000) * full_price
output_cost = (output_tokens / 1_000_000) * output_price
total = cached_cost + uncached_cost + output_cost
```

## Practical tips
1. Keep system messages stable across all turns
2. Never modify old messages - only append new ones
3. For long conversations, caching saves 50-90% on input tokens
4. Design agent loops to maximize cache hits
5. Monitor cache hit rates in production
6. Consider cache expiration when designing user interactions

## Files created
- `experiments/03_caching_impact.py` — caching cost comparison
- `notes/day3_caching.md` — this file
