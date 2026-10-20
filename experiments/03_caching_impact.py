"""
Day 3 Experiment: Prompt caching impact on costs.

We'll simulate a multi-turn conversation and calculate:
1. Costs WITHOUT caching (baseline)
2. Costs WITH caching (using different provider discounts)
3. What breaks the cache
4. Real-world savings for agent loops
"""

import tiktoken

# --- Pricing configurations (USD per 1M tokens) ---
# These are approximate prices as of early 2025

PRICING = {
    "anthropic_claude": {
        "name": "Anthropic Claude 3.5 Sonnet",
        "input_price": 3.00,           # $3/1M input tokens
        "cached_price": 0.30,          # 90% discount = $0.30/1M
        "output_price": 15.00,         # $15/1M output tokens
        "cache_storage_per_hour": 3.75,  # $3.75/1M tokens/hour
        "min_cache_tokens": 1024,
        "cache_ttl_minutes": 5,
    },
    "openai_gpt4o": {
        "name": "OpenAI GPT-4o",
        "input_price": 2.50,           # $2.50/1M input tokens
        "cached_price": 1.25,          # 50% discount = $1.25/1M
        "output_price": 10.00,         # $10/1M output tokens
        "min_cache_tokens": 1024,
        "cache_ttl_minutes": 60,
    },
    "no_caching": {
        "name": "No caching (baseline)",
        "input_price": 3.00,           # Same as Anthropic for fair comparison
        "cached_price": 3.00,          # No discount
        "output_price": 15.00,
        "min_cache_tokens": 0,
        "cache_ttl_minutes": 0,
    },
}


def count_tokens(messages):
    """Count tokens in a list of chat messages using cl100k_base encoding."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 4  # Overhead per message
    
    total = 0
    for msg in messages:
        total += tokens_per_message
        total += len(enc.encode(msg["content"]))
    
    total += 3  # Assistant reply priming
    return total


def calculate_cost_without_caching(input_tokens, output_tokens, pricing):
    """Calculate cost when every token is charged at full price."""
    input_cost = (input_tokens / 1_000_000) * pricing["input_price"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_price"]
    return input_cost + output_cost


def calculate_cost_with_caching(input_tokens, output_tokens, cached_tokens, pricing):
    """
    Calculate cost with caching applied.
    
    cached_tokens: number of tokens that hit the cache (get discount)
    uncached_tokens: input_tokens - cached_tokens (full price)
    """
    uncached_tokens = input_tokens - cached_tokens
    
    # Ensure we don't go negative
    uncached_tokens = max(0, uncached_tokens)
    
    cached_cost = (cached_tokens / 1_000_000) * pricing["cached_price"]
    uncached_cost = (uncached_tokens / 1_000_000) * pricing["input_price"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_price"]
    
    return cached_cost + uncached_cost + output_cost


def simulate_conversation_with_caching():
    """
    Simulate a 10-turn coding agent conversation and compare costs
    with and without caching.
    """
    print("=" * 70)
    print("PART 1: 10-Turn Coding Agent - Caching Impact")
    print("=" * 70)
    
    # Simulate a coding agent conversation
    system_msg = {
        "role": "system",
        "content": "You are a Python coding assistant. Help debug code, explain errors, and suggest improvements. Be concise."
    }
    
    # Simulated conversation turns (user question + assistant response)
    turns = [
        ("Why is my list empty after the loop?", 
         "You're likely iterating over the list while modifying it. Use a copy: for item in list[:]."),
        ("Show me an example",
         "items = [1,2,3]\nfor item in items[:]:\n    if item == 2:\n        items.remove(item)\nprint(items)  # [1, 3]"),
        ("What about dictionaries?",
         "Same issue. Iterate over list(d.keys()) if you need to delete keys during iteration."),
        ("How do I handle KeyError?",
         "Use dict.get(key, default) or check with 'if key in dict' before accessing."),
        ("What's the difference between .get() and .setdefault()?",
         ".get() returns default without modifying dict. .setdefault() inserts the default if key is missing."),
        ("Show me both",
         "d = {}\nval1 = d.get('x', 0)  # val1=0, d still empty\nval2 = d.setdefault('y', 0)  # val2=0, d={'y':0}"),
        ("Which is faster for large dicts?",
         "Both are O(1). .get() is slightly faster if you don't need to insert the default."),
        ("How do I merge two dicts?",
         "Python 3.9+: d1 | d2. Older: {**d1, **d2} or d1.update(d2)."),
        ("What about nested dicts?",
         "Use a loop or collections.defaultdict. For deep merge, write a recursive function."),
        ("Thanks, that's all",
         "You're welcome! Happy coding."),
    ]
    
    messages = [system_msg]
    
    # Track costs for each provider
    costs = {
        "no_caching": {"total": 0.0, "input_tokens": 0, "cached_tokens": 0},
        "anthropic_claude": {"total": 0.0, "input_tokens": 0, "cached_tokens": 0},
        "openai_gpt4o": {"total": 0.0, "input_tokens": 0, "cached_tokens": 0},
    }
    
    print(f"\nSystem message: {count_tokens([system_msg])} tokens")
    print(f"\n{'Turn':<6} {'Input':<10} {'Output':<10} {'No Cache':<12} {'Anthropic':<12} {'OpenAI':<12}")
    print("-" * 70)
    
    for i in range(0, len(turns), 2):
        # Add user message
        user_content = turns[i][0]
        messages.append({"role": "user", "content": user_content})
        
        # Count input tokens
        input_tokens = count_tokens(messages)
        
        # Add assistant response
        asst_content = turns[i + 1][0]
        messages.append({"role": "assistant", "content": asst_content})
        
        # Count output tokens
        output_tokens = len(tiktoken.get_encoding("cl100k_base").encode(asst_content))
        
        # Determine cached tokens (everything except the new user message)
        # In reality, the cache includes system + all previous messages
        # For simplicity, we'll cache everything except the last user message
        user_msg_tokens = count_tokens([{"role": "user", "content": user_content}])
        cached_tokens = input_tokens - user_msg_tokens
        
        # Calculate costs for each provider
        turn_no_cache = calculate_cost_without_caching(
            input_tokens, output_tokens, PRICING["no_caching"]
        )
        turn_anthropic = calculate_cost_with_caching(
            input_tokens, output_tokens, cached_tokens, PRICING["anthropic_claude"]
        )
        turn_openai = calculate_cost_with_caching(
            input_tokens, output_tokens, cached_tokens, PRICING["openai_gpt4o"]
        )
        
        # Update totals
        costs["no_caching"]["total"] += turn_no_cache
        costs["anthropic_claude"]["total"] += turn_anthropic
        costs["openai_gpt4o"]["total"] += turn_openai
        costs["no_caching"]["input_tokens"] += input_tokens
        costs["anthropic_claude"]["input_tokens"] += input_tokens
        costs["anthropic_claude"]["cached_tokens"] += cached_tokens
        costs["openai_gpt4o"]["input_tokens"] += input_tokens
        costs["openai_gpt4o"]["cached_tokens"] += cached_tokens
        
        turn_num = (i // 2) + 1
        print(f"{turn_num:<6} {input_tokens:<10} {output_tokens:<10} ${turn_no_cache:<11.6f} ${turn_anthropic:<11.6f} ${turn_openai:<11.6f}")
    
    print("-" * 70)
    
    # Summary
    print(f"\nTotal input tokens: {costs['no_caching']['input_tokens']:,}")
    print(f"Cached tokens (Anthropic/OpenAI): {costs['anthropic_claude']['cached_tokens']:,}")
    print(f"\nTotal costs:")
    print(f"  No caching:        ${costs['no_caching']['total']:.6f}")
    print(f"  Anthropic (90%):   ${costs['anthropic_claude']['total']:.6f}")
    print(f"  OpenAI (50%):      ${costs['openai_gpt4o']['total']:.6f}")
    
    savings_anthropic = (1 - costs['anthropic_claude']['total'] / costs['no_caching']['total']) * 100
    savings_openai = (1 - costs['openai_gpt4o']['total'] / costs['no_caching']['total']) * 100
    
    print(f"\nSavings:")
    print(f"  Anthropic: {savings_anthropic:.1f}%")
    print(f"  OpenAI:    {savings_openai:.1f}%")


def demonstrate_cache_breaking():
    """Show what breaks the cache."""
    print("\n\n" + "=" * 70)
    print("PART 2: What Breaks the Cache")
    print("=" * 70)
    
    print("\nThe cache requires an EXACT prefix match. Even small changes break it:")
    
    scenarios = [
        ("Same system message", True),
        ("Changed system message (added one word)", False),
        ("Reordered conversation history", False),
        ("Added whitespace to system message", False),
        ("Changed one character in old message", False),
    ]
    
    print(f"\n{'Scenario':<45} {'Cache Hit?':<12}")
    print("-" * 70)
    for scenario, hits in scenarios:
        status = "YES" if hits else "NO - full price!"
        print(f"{scenario:<45} {status:<12}")
    
    print("\nKey insight: Keep the system message and conversation history STABLE.")
    print("Only add new messages at the END, never modify old ones.")


def demonstrate_minimum_cache_size():
    """Show the impact of minimum cache size requirements."""
    print("\n\n" + "=" * 70)
    print("PART 3: Minimum Cache Size Impact")
    print("=" * 70)
    
    print("\nMost providers require at least 1024 tokens to cache.")
    print("Short conversations don't benefit from caching:\n")
    
    # Simulate short vs long conversations
    conversations = [
        ("Short (500 tokens)", 500),
        ("Medium (1500 tokens)", 1500),
        ("Long (5000 tokens)", 5000),
    ]
    
    print(f"{'Conversation':<25} {'Input Tokens':<15} {'Cacheable?':<12} {'Savings':<12}")
    print("-" * 70)
    
    for name, tokens in conversations:
        cacheable = tokens >= 1024
        savings = "90% on cached portion" if cacheable else "No caching available"
        print(f"{name:<25} {tokens:<15} {'YES' if cacheable else 'NO':<12} {savings:<12}")
    
    print("\nFor agents: Early turns won't cache, but later turns (1000+ tokens) will.")


def main():
    print("\n" + "=" * 70)
    print("DAY 3 EXPERIMENT: Prompt Caching Impact on Costs")
    print("=" * 70)
    
    simulate_conversation_with_caching()
    demonstrate_cache_breaking()
    demonstrate_minimum_cache_size()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Caching reduces costs by 50-90% for repeated context")
    print("2. Essential for agent loops where context snowballs")
    print("3. Cache requires exact prefix match - don't modify old messages")
    print("4. Minimum cache size (1024 tokens) means short conversations don't benefit")
    print("5. Cache expires after 5-60 minutes of inactivity")
    print("6. Production agents almost always use caching - it's not optional")
    print("=" * 70)


if __name__ == "__main__":
    main()
