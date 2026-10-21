# Day 4: Model Comparison Framework

## What I learned

### Why compare models?
- Different models excel at different tasks
- Cost varies dramatically (GPT-4 is 20x more expensive than GPT-3.5)
- Latency matters for user experience
- Context limits affect what you can build
- Quality varies by task type

### What to measure

**Latency:**
- **TTFT (Time To First Token)**: How long until response starts?
- **TPS (Tokens Per Second)**: How fast does it generate?
- **Total latency**: End-to-end time

**Token usage:**
- Input tokens (prompt + context)
- Output tokens (response length)
- Tokenization efficiency (same text, different counts)

**Cost:**
- Cost per request
- Cost per successful task (quality-adjusted)
- Monthly estimate for your use case

**Quality:**
- Task success rate
- Accuracy (for factual tasks)
- Code correctness (for coding)
- Human evaluation (for subjective tasks)

### Cloud vs local tradeoffs

| Factor | Cloud | Local |
|--------|-------|-------|
| Cost | Pay per token | Free (your hardware) |
| Speed | Fast (powerful GPUs) | Slower (limited VRAM) |
| Quality | State-of-the-art | Good for small models |
| Privacy | Data sent to provider | Data stays local |
| Context | 128K-1M tokens | 4K-32K (VRAM limited) |
| Setup | API key | Install + download model |

### Your hardware (GTX 1650, 4GB VRAM)
Can run:
- **Llama 3.2 1B** (quantized): Fast, decent quality
- **Phi-3 mini** (3.8B, quantized): Better quality, slower
- **Qwen 3.5 4B**: Good balance (you have this)
- **Mistral 7B** (quantized): Good quality, may be slow

### Model comparison results (from experiment)

**Local model (qwen3.5:4b):**
- Cost: $0 (your hardware)
- TTFT: ~0.5-1.0s
- TPS: ~15-25 tokens/sec
- Quality: Good for basic tasks

**Cloud models (mock data):**
- GPT-4: Fast (TTFT 0.3s, TPS 50), expensive ($10/1M input)
- Claude 3.5: Fast, cheaper than GPT-4 ($3/1M input), 90% caching

### Key insights for AI engineering

1. **Test on real tasks** - benchmarks don't tell the whole story
2. **Measure quality + cost** - cheapest isn't always best
3. **Consider latency** - users notice slow responses
4. **Local for dev, cloud for prod** - iterate locally, deploy to cloud
5. **Cache when possible** - 50-90% savings on repeated context
6. **Choose per task** - use GPT-4 for hard reasoning, GPT-3.5 for simple tasks

## Vocabulary
- **TTFT**: Time To First Token - latency until response starts
- **TPS**: Tokens Per Second - generation speed
- **Quantization**: Compressing model weights (INT8, INT4) to fit in VRAM
- **VRAM**: Video RAM - GPU memory for model weights + activations
- **Benchmark**: Standardized test for model comparison

## Code patterns

```python
# Measure latency
start = time.time()
response = call_model(prompt)
ttft = time_to_first_token - start
total_time = time.time() - start
tps = output_tokens / (total_time - ttft)

# Calculate cost
input_cost = (input_tokens / 1_000_000) * price_per_million_input
output_cost = (output_tokens / 1_000_000) * price_per_million_output
total_cost = input_cost + output_cost

# Compare models
for model in models:
    results = test_model(model, tasks)
    print(f"{model}: avg TPS={avg_tps}, cost=${total_cost}")
```

## Framework structure

The experiment builds a reusable framework:
1. Define tasks (coding, reasoning, Q&A, summarization)
2. Measure latency and tokens for each model
3. Calculate costs
4. Generate comparison report

**Extending it:**
- Add real API calls (OpenAI, Anthropic, Google)
- Add quality metrics (accuracy, success rate)
- Add more tasks
- Add visualization (charts, graphs)

## Files created
- `experiments/04_model_comparison.py` — comparison framework
- `notes/day4_comparison.md` — this file
