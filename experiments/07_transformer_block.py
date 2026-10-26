"""
Day 7: Complete Transformer Block and Text Generation

Demonstrates:
- Feed-forward networks
- Layer normalization
- Residual connections
- Complete transformer block flow
- Autoregressive text generation
"""

import numpy as np

def layer_norm(x, epsilon=1e-5):
    """Normalize across feature dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

def feed_forward(x, w1, w2):
    """Two-layer feed-forward network with ReLU activation."""
    hidden = np.maximum(0, np.dot(x, w1))  # ReLU
    return np.dot(hidden, w2)

def transformer_block(x, attention_output, ff_w1, ff_w2):
    """
    Complete transformer block:
    1. Add residual connection after attention
    2. Layer normalization
    3. Feed-forward network
    4. Add residual connection after FFN
    5. Layer normalization
    """
    # Post-attention: residual + norm
    x = x + attention_output
    x = layer_norm(x)
    
    # Feed-forward
    ff_output = feed_forward(x, ff_w1, ff_w2)
    
    # Post-FFN: residual + norm
    x = x + ff_output
    x = layer_norm(x)
    
    return x

def demonstrate_transformer_block():
    """Show how a complete transformer block processes tokens."""
    
    print("=" * 70)
    print("DAY 7: COMPLETE TRANSFORMER BLOCK")
    print("=" * 70)
    
    # Simulate 3 tokens with 4-dimensional embeddings
    tokens = ["The", "cat", "sat"]
    seq_len = len(tokens)
    d_model = 4
    
    print(f"\nInput: {tokens}")
    print(f"Sequence length: {seq_len} tokens")
    print(f"Embedding dimension: {d_model}")
    
    # Initial embeddings (from Day 6 attention output)
    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)
    print(f"\nInitial embeddings shape: {x.shape}")
    print(f"Sample values (token 0): {x[0].round(3)}")
    
    # Simulate attention output (from Day 6)
    attention_output = np.random.randn(seq_len, d_model) * 0.1
    print(f"\nAttention output (small adjustments): {attention_output[0].round(3)}")
    
    # Feed-forward weights
    d_ff = 8  # FFN hidden dimension (usually 4x d_model)
    ff_w1 = np.random.randn(d_model, d_ff)
    ff_w2 = np.random.randn(d_ff, d_model)
    
    print(f"\n--- Transformer Block Processing ---")
    
    # Step 1: Residual connection after attention
    step1 = x + attention_output
    print(f"\n1. After residual connection (x + attention):")
    print(f"   Token 0: {step1[0].round(3)}")
    
    # Step 2: Layer normalization
    step2 = layer_norm(step1)
    print(f"\n2. After layer normalization:")
    print(f"   Token 0: {step2[0].round(3)}")
    print(f"   Mean: {step2[0].mean():.3f}, Std: {step2[0].std():.3f}")
    
    # Step 3: Feed-forward network
    ff_output = feed_forward(step2, ff_w1, ff_w2)
    print(f"\n3. After feed-forward network:")
    print(f"   Token 0: {ff_output[0].round(3)}")
    
    # Step 4: Residual connection after FFN
    step4 = step2 + ff_output
    print(f"\n4. After residual connection (norm + FFN output):")
    print(f"   Token 0: {step4[0].round(3)}")
    
    # Step 5: Final layer normalization
    output = layer_norm(step4)
    print(f"\n5. Final output (after second norm):")
    print(f"   Token 0: {output[0].round(3)}")
    print(f"   Mean: {output[0].mean():.3f}, Std: {output[0].std():.3f}")
    
    print(f"\nOutput shape: {output.shape} (same as input)")
    
    # Demonstrate autoregressive generation
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE TEXT GENERATION")
    print("=" * 70)
    
    print("\nHow models generate text token-by-token:")
    print("\nStep 1: Input 'The' -> Model predicts next token probabilities")
    print("        P('cat')=0.4, P('dog')=0.3, P('bird')=0.1, ...")
    print("        Sample: 'cat'")
    
    print("\nStep 2: Input 'The cat' -> Model predicts next token")
    print("        P('sat')=0.5, P('ran')=0.2, P('jumped')=0.15, ...")
    print("        Sample: 'sat'")
    
    print("\nStep 3: Input 'The cat sat' -> Model predicts next token")
    print("        P('on')=0.6, P('down')=0.2, P('here')=0.1, ...")
    print("        Sample: 'on'")
    
    print("\nThis continues until:")
    print("  - Model generates [EOS] (end of sequence) token")
    print("  - Maximum length reached")
    print("  - Stop condition met")
    
    # Show why this is slow
    print("\n" + "=" * 70)
    print("WHY GENERATION IS SLOW")
    print("=" * 70)
    
    print("\nEach new token requires:")
    print("  1. Process entire sequence through all transformer layers")
    print("  2. Compute attention for all token pairs")
    print("  3. Sample next token from probability distribution")
    print("  4. Append token and repeat")
    
    print("\nFor a 100-token response with 12-layer model:")
    print("  - 100 sequential steps (can't parallelize)")
    print("  - Each step processes 1 to 100 tokens")
    print("  - Total: ~5000 token-passes through 12 layers")
    print("  - This is why generation is slower than processing input")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Transformer block = Attention + Residual + Norm + FFN + Residual + Norm")
    print("2. Residual connections prevent vanishing gradients in deep networks")
    print("3. Layer normalization stabilizes training and improves convergence")
    print("4. Feed-forward networks add non-linear transformation capacity")
    print("5. Text generation is autoregressive: each token depends on all previous")
    print("6. Generation is slow because it's inherently sequential")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_transformer_block()
