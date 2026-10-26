# Day 6: Transformer Architecture and Self-Attention

## What is a Transformer?

A transformer is the neural network architecture behind modern LLMs. Unlike older models that process text sequentially, transformers process all tokens simultaneously and learn relationships between them.

**Key insight:** Transformers don't read left-to-right. They see the entire sequence at once and figure out which tokens relate to each other.

## Self-Attention: The Core Mechanism

Self-attention lets each token "pay attention" to every other token in the sequence.

**Intuition:** When you read "The animal didn't cross the street because it was too tired," you understand "it" refers to "animal" not "street." Self-attention learns these connections automatically.

**Technically:** For each token, the model computes three vectors:
- **Query (Q):** What this token is looking for
- **Key (K):** What this token offers to others  
- **Value (V):** The actual information this token carries

The attention score between two tokens is:
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

Where:
- `Q × K^T` computes similarity between query and all keys
- `/ √d` scales by dimension to prevent vanishing gradients
- `softmax` converts scores to probabilities
- `× V` produces weighted sum of values

## Multi-Head Attention

Instead of one attention calculation, transformers use multiple "heads" — each learning different types of relationships:
- Head 1 might focus on syntax (subject-verb agreement)
- Head 2 might focus on semantics (pronoun resolution)
- Head 3 might focus on position (word order)
- Head 4 might focus on long-range dependencies

## Computational Complexity

Attention is **O(n²)** where n is sequence length:
- 10 tokens → 100 attention pairs
- 100 tokens → 10,000 pairs
- 1024 tokens → 1,048,576 pairs
- 8192 tokens → 67,108,864 pairs

This is why:
- Longer contexts cost more to process
- Context windows have practical limits
- Efficient attention variants exist (sparse, linear, etc.)

## Why This Matters for AI Engineering

1. **Token efficiency matters:** Fewer tokens = less computation = lower cost
2. **Context window limits are real:** Not just arbitrary caps, but computational constraints
3. **Different models tokenize differently:** Affects attention patterns and efficiency
4. **Long documents need special handling:** Chunking, summarization, or efficient attention

## Experiment Results

Run `experiments/06_attention_visualization.py` to see:
- How attention weights distribute across tokens
- Why computational cost grows quadratically
- How different attention heads focus on different relationships

## Key Terms

- **Self-attention:** Mechanism for tokens to relate to each other
- **Query/Key/Value:** Three vectors computed per token for attention
- **Attention head:** One instance of attention computation
- **Multi-head attention:** Multiple parallel attention computations
- **O(n²):** Quadratic complexity — cost grows with square of input size

## Files Created

- `experiments/06_attention_visualization.py` — attention computation demo
- `notes/day6_transformers.md` — this file
