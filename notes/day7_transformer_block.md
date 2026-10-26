# Day 7: Complete Transformer Block and Text Generation

## Transformer Block Components

A complete transformer block has 6 steps:

1. **Self-Attention** (Day 6): Tokens attend to each other
2. **Residual Connection**: Add attention output to input
3. **Layer Normalization**: Normalize across features
4. **Feed-Forward Network**: Two-layer neural network with ReLU
5. **Residual Connection**: Add FFN output to normalized input
6. **Layer Normalization**: Final normalization

## Why Residual Connections?

Deep neural networks suffer from **vanishing gradients**: as gradients backpropagate through many layers, they shrink exponentially, making early layers learn slowly.

Residual connections solve this by creating "skip connections":
```
output = input + F(input)
```

The gradient can flow directly through the `+ input` path, bypassing the transformation `F()`. This allows training very deep networks (GPT-3 has 96 layers).

## Why Layer Normalization?

Layer normalization stabilizes training by:
- Centering activations around zero mean
- Scaling to unit variance
- Reducing internal covariate shift

Formula:
```
norm(x) = (x - mean) / (std + ε)
```

Without normalization, activations can explode or vanish, making training unstable.

## Feed-Forward Networks

Each transformer block has a two-layer FFN:
```
FFN(x) = ReLU(x × W1) × W2
```

- **W1**: Expands dimension (typically 4x model dimension)
- **ReLU**: Adds non-linearity
- **W2**: Projects back to model dimension

The FFN adds capacity for complex transformations that attention alone can't capture.

## Autoregressive Generation

Language models generate text one token at a time:

1. Input: "The cat"
2. Model computes probabilities for next token
3. Sample from distribution: P("sat")=0.5, P("ran")=0.3, ...
4. Append sampled token: "The cat sat"
5. Repeat until [EOS] or max length

**Why this is slow:**
- Each token requires processing entire sequence
- Can't parallelize (token N depends on tokens 1 to N-1)
- 100-token response = 100 sequential model passes
- Each pass computes attention for all token pairs

**Contrast with input processing:**
- Input tokens are processed in parallel
- All attention computations happen simultaneously
- Much faster than generation

## Computational Flow

For a sequence of length n with d-dimensional embeddings:

```
Input: [n × d]
  ↓
Self-Attention: [n × d] → [n × d] (O(n² × d) computation)
  ↓
Residual + Norm: [n × d] → [n × d]
  ↓
Feed-Forward: [n × d] → [n × 4d] → [n × d]
  ↓
Residual + Norm: [n × d] → [n × d]
  ↓
Output: [n × d]
```

A typical model has 12-96 such blocks stacked sequentially.

## Key Insights for AI Engineering

1. **Depth matters**: More transformer blocks = more capacity but slower inference
2. **Width matters**: Larger d_model = richer representations but more memory
3. **Generation is the bottleneck**: Input processing is parallel, generation is sequential
4. **KV cache helps**: Store key/value vectors to avoid recomputing attention for previous tokens
5. **Batching helps**: Process multiple sequences simultaneously during training

## Experiment Results

Run `experiments/07_transformer_block.py` to see:
- Complete transformer block processing step-by-step
- How residual connections preserve information
- How layer normalization stabilizes values
- Why autoregressive generation is inherently sequential

## Files Created

- `experiments/07_transformer_block.py` — transformer block and generation demo
- `notes/day7_transformer_block.md` — this file
