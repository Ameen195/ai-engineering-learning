"""
Day 2 Experiment: Context windows and token counting.

We'll:
1. Count tokens accurately using the model-appropriate tokenizer
2. Simulate a conversation and watch context grow
3. Estimate costs based on token counts
4. See what happens when we approach context limits
"""

import tiktoken

# --- Configuration ---
# Using GPT-4 pricing as of early 2024 (approximate, in USD per 1M tokens)
GPT4_INPUT_PRICE = 10.00   # $10 per 1M input tokens
GPT4_OUTPUT_PRICE = 30.00  # $30 per 1M output tokens
GPT4_CONTEXT_LIMIT = 128000  # 128K tokens

# For comparison: GPT-3.5-turbo
GPT35_INPUT_PRICE = 0.50
GPT35_OUTPUT_PRICE = 1.50
GPT35_CONTEXT_LIMIT = 16385  # ~16K tokens


def count_tokens(messages, model="gpt-4"):
    """
    Count tokens in a list of chat messages.
    
    Important: Chat models add overhead tokens for message formatting.
    This is an approximation based on OpenAI's documentation.
    """
    if model.startswith("gpt-4"):
        enc = tiktoken.encoding_for_model("gpt-4")
    else:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    # Each message has overhead tokens (role, formatting, etc.)
    # Approximation: 4 tokens per message for message formatting
    tokens_per_message = 4
    
    total = 0
    for msg in messages:
        total += tokens_per_message
        total += len(enc.encode(msg["content"]))
    
    # Add 3 tokens for the assistant reply priming (start of response)
    total += 3
    
    return total


def estimate_cost(input_tokens, output_tokens, model="gpt-4"):
    """Estimate cost in USD based on token counts and model pricing."""
    if model.startswith("gpt-4"):
        input_cost = (input_tokens / 1_000_000) * GPT4_INPUT_PRICE
        output_cost = (output_tokens / 1_000_000) * GPT4_OUTPUT_PRICE
    else:
        input_cost = (input_tokens / 1_000_000) * GPT35_INPUT_PRICE
        output_cost = (output_tokens / 1_000_000) * GPT35_OUTPUT_PRICE
    
    return input_cost + output_cost


def simulate_conversation():
    """
    Simulate a multi-turn conversation and track how context grows.
    This demonstrates the "context snowball" effect.
    """
    print("=" * 70)
    print("PART 1: Context Growth in a Conversation")
    print("=" * 70)
    
    # Start with a system message
    system_msg = {
        "role": "system",
        "content": "You are a helpful Python programming assistant. Explain concepts clearly with examples."
    }
    
    messages = [system_msg]
    
    # Simulate conversation turns
    turns = [
        ("user", "What is a list comprehension?"),
        ("assistant", "A list comprehension is a concise way to create lists in Python. Example: squares = [x**2 for x in range(10)]"),
        ("user", "Can you show me a more complex example?"),
        ("assistant", "Sure! Here's filtering even numbers: evens = [x for x in range(20) if x % 2 == 0]. You can also nest them: matrix = [[i*j for j in range(5)] for i in range(5)]"),
        ("user", "How do I use it with dictionaries?"),
        ("assistant", "Dictionary comprehensions work similarly: word_lengths = {word: len(word) for word in ['hello', 'world', 'python']}. You can also filter: long_words = {w: len(w) for w in words if len(w) > 3}"),
        ("user", "What about sets?"),
        ("assistant", "Set comprehensions: unique_lengths = {len(word) for word in ['apple', 'banana', 'apple']}. Note the curly braces without key-value pairs. Sets automatically remove duplicates."),
        ("user", "Are there performance considerations?"),
        ("assistant", "Yes! List comprehensions are generally faster than equivalent for-loops because they're optimized in C. However, for very large datasets, consider generator expressions: (x**2 for x in range(1000000)) uses less memory because it yields items lazily instead of creating the full list in memory."),
    ]
    
    print(f"\nSystem message: {len(tiktoken.get_encoding('cl100k_base').encode(system_msg['content']))} tokens")
    print(f"\n{'Turn':<6} {'Input Tokens':<15} {'Output Tokens':<15} {'Total Input':<15} {'Cost (USD)':<12}")
    print("-" * 70)
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for i in range(0, len(turns), 2):
        # Add user message
        user_role, user_content = turns[i]
        messages.append({"role": user_role, "content": user_content})
        
        # Count input tokens (everything before the response)
        input_tokens = count_tokens(messages, model="gpt-4")
        
        # Add assistant response
        asst_role, asst_content = turns[i + 1]
        messages.append({"role": asst_role, "content": asst_content})
        
        # Count output tokens (just the response)
        output_tokens = len(tiktoken.get_encoding('cl100k_base').encode(asst_content))
        
        # Calculate cost for this turn
        turn_cost = estimate_cost(input_tokens, output_tokens, model="gpt-4")
        
        # Update totals
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += turn_cost
        
        turn_num = (i // 2) + 1
        print(f"{turn_num:<6} {input_tokens:<15} {output_tokens:<15} {total_input_tokens:<15} ${turn_cost:<11.6f}")
    
    print("-" * 70)
    print(f"\nTotal input tokens consumed: {total_input_tokens:,}")
    print(f"Total output tokens generated: {total_output_tokens:,}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"\nNotice: Input tokens grow with each turn because we re-send the entire history!")
    print(f"Final context size: {count_tokens(messages, model='gpt-4'):,} tokens")


def demonstrate_context_limits():
    """Show what happens when we approach context limits."""
    print("\n\n" + "=" * 70)
    print("PART 2: Context Window Limits")
    print("=" * 70)
    
    # Create a large document
    large_text = "Python is a programming language. " * 1000  # ~5000 tokens
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Summarize this text:\n\n{large_text}"}
    ]
    
    input_tokens = count_tokens(messages, model="gpt-4")
    
    print(f"\nLarge document token count: {input_tokens:,}")
    print(f"GPT-4 context limit: {GPT4_CONTEXT_LIMIT:,}")
    print(f"Remaining for response: {GPT4_CONTEXT_LIMIT - input_tokens:,}")
    print(f"Usage: {(input_tokens / GPT4_CONTEXT_LIMIT * 100):.1f}% of context window")
    
    # Now try with GPT-3.5-turbo
    input_tokens_35 = count_tokens(messages, model="gpt-3.5-turbo")
    print(f"\nSame document with GPT-3.5-turbo: {input_tokens_35:,} tokens")
    print(f"GPT-3.5-turbo context limit: {GPT35_CONTEXT_LIMIT:,}")
    
    if input_tokens_35 > GPT35_CONTEXT_LIMIT:
        print(f"ERROR: Document exceeds GPT-3.5-turbo context by {input_tokens_35 - GPT35_CONTEXT_LIMIT:,} tokens!")
        print("This request would fail or require truncation.")
    else:
        print(f"Remaining for response: {GPT35_CONTEXT_LIMIT - input_tokens_35:,}")


def compare_model_costs():
    """Compare costs across models for the same task."""
    print("\n\n" + "=" * 70)
    print("PART 3: Cost Comparison Across Models")
    print("=" * 70)
    
    messages = [
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "Explain recursion with a simple example."},
    ]
    
    input_tokens = count_tokens(messages, model="gpt-4")
    estimated_output = 150  # Typical response length
    
    gpt4_cost = estimate_cost(input_tokens, estimated_output, model="gpt-4")
    gpt35_cost = estimate_cost(input_tokens, estimated_output, model="gpt-3.5-turbo")
    
    print(f"\nPrompt: {input_tokens} input tokens, ~{estimated_output} output tokens")
    print(f"\nGPT-4 cost:        ${gpt4_cost:.6f}")
    print(f"GPT-3.5-turbo cost: ${gpt35_cost:.6f}")
    print(f"\nGPT-4 is {gpt4_cost / gpt35_cost:.1f}x more expensive for the same prompt!")
    print(f"\nBut context limits differ:")
    print(f"  GPT-4: {GPT4_CONTEXT_LIMIT:,} tokens")
    print(f"  GPT-3.5: {GPT35_CONTEXT_LIMIT:,} tokens")


def main():
    print("\n" + "=" * 70)
    print("DAY 2 EXPERIMENT: Context Windows and Token Counting")
    print("=" * 70)
    
    simulate_conversation()
    demonstrate_context_limits()
    compare_model_costs()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Context grows with each turn because history is re-sent")
    print("2. Input tokens are cumulative - this is why agents are expensive")
    print("3. Different models have different context limits and pricing")
    print("4. Always count tokens BEFORE calling the API to avoid surprises")
    print("5. Use the correct tokenizer for the model you're calling")
    print("=" * 70)


if __name__ == "__main__":
    main()
