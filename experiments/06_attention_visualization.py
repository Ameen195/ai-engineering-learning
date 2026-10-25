"""
Day 6: Self-Attention Visualization

Demonstrates how transformers compute attention between tokens.
Shows why context window size affects computational cost.
"""

import numpy as np

def softmax(x):
    """Convert raw scores to probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_attention(query, keys, values):
    """
    Compute attention scores between a query and all keys.
    Returns weighted sum of values.
    """
    scores = np.dot(query, keys.T) / np.sqrt(query.shape[-1])
    weights = softmax(scores)
    return weights, np.dot(weights, values)

def demonstrate_attention():
    """Show how attention works on a simple sentence."""
    
    print("=" * 70)
    print("DAY 6: SELF-ATTENTION VISUALIZATION")
    print("=" * 70)
    
    # Simple example: "The cat sat on the mat"
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    print(f"\nInput tokens: {tokens}")
    print(f"Sequence length: {len(tokens)} tokens")
    
    # Simulate embeddings (6 tokens x 4 dimensions)
    np.random.seed(42)
    embeddings = np.random.randn(len(tokens), 4)
    
    print("\n--- Attention from 'cat' (token 1) to all other tokens ---")
    query = embeddings[1]  # "cat"
    keys = embeddings
    values = embeddings
    
    weights, output = compute_attention(query, keys, values)
    
    print("\nAttention weights (how much 'cat' attends to each token):")
    for i, (token, weight) in enumerate(zip(tokens, weights)):
        bar = "#" * int(weight * 20)
        print(f"  {token:6s} [{i}]: {weight:.3f} {bar}")
    
    print(f"\nOutput vector shape: {output.shape}")
    print(f"Output (first 3 dims): {output[:3].round(3)}")
    
    # Show computational complexity
    print("\n" + "=" * 70)
    print("COMPUTATIONAL COMPLEXITY")
    print("=" * 70)
    
    seq_lengths = [10, 100, 512, 1024, 4096, 8192]
    print(f"\n{'Seq Length':<12} {'Attention Pairs':<20} {'Relative Cost':<15}")
    print("-" * 50)
    base_cost = 10 * 10
    for n in seq_lengths:
        pairs = n * n
        relative = pairs / base_cost
        print(f"{n:<12} {pairs:<20,} {relative:<15.1f}x")
    
    print("\nKey insight: Attention is O(n²) — doubling context quadruples cost.")
    
    # Multi-head attention demo
    print("\n" + "=" * 70)
    print("MULTI-HEAD ATTENTION")
    print("=" * 70)
    
    num_heads = 4
    print(f"\nUsing {num_heads} attention heads on '{' '.join(tokens)}'")
    print("\nEach head learns different relationships:")
    
    for head in range(num_heads):
        np.random.seed(head * 100)
        query = np.random.randn(4)
        keys = np.random.randn(len(tokens), 4)
        weights, _ = compute_attention(query, keys, np.random.randn(len(tokens), 4))
        
        top_token = tokens[np.argmax(weights)]
        print(f"  Head {head+1}: strongest attention on '{top_token}' (weight: {np.max(weights):.3f})")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Self-attention lets each token relate to every other token")
    print("2. Computational cost grows quadratically with sequence length")
    print("3. Multiple attention heads capture different relationship types")
    print("4. This is why context windows have practical limits")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_attention()
