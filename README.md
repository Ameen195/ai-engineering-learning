# AI Engineering Learning Lab

A hands-on learning project for understanding how language models work internally while building practical AI engineering skills.

## What This Is

I'm building small, focused experiments to master:
- How tokens and tokenization work across different models
- Context windows, prompt design, and cost optimization
- Transformer architecture and model internals
- Building AI agents with tool calling and RAG

Each day focuses on one concept with runnable Python experiments.

## Current Progress

### Day 1: Tokens and Tokenization
- **Experiment**: `experiments/01_tokenize_and_count.py`
- **Concepts**: What tokens are, how tokenizers differ, why token counts vary
- **Key insight**: Same text = different token counts across models. You can't compare token counts directly.
- **Notes**: `notes/day1_tokens.md`

### Day 2: Context Windows and Token Counting
- **Experiment**: `experiments/02_context_and_counting.py`
- **Concepts**: Context limits, input vs output tokens, the context snowball effect
- **Key insight**: Conversation history grows cumulatively in agent loops — this is the main cost driver.
- **Notes**: `notes/day2_context.md`

### Day 3: Prompt Caching and Optimization
- **Experiment**: `experiments/03_caching_impact.py`
- **Concepts**: Prompt caching, cache hits/misses, cost optimization for agent loops
- **Key insight**: Caching reduces costs by 50-90% for repeated context. Essential for production agents where context snowballs.
- **Notes**: `notes/day3_caching.md`

### Day 4: Model Comparison Framework
- **Experiment**: `experiments/04_model_comparison_demo.py`
- **Concepts**: Comparing models on latency, cost, and token efficiency
- **Note**: The demo uses mock values for latency, token counts, and costs; it is not a real model benchmark.
- **Notes**: `notes/day4_comparison.md`

### Day 5: Python Code Review Learning Project
- **Static web demo**: `public/index.html` and `public/review.js` run three transparent pattern checks in your browser
- **No web API or model call**: source is not uploaded; the deployed static demo needs no Groq key or password
- **Limitations**: heuristics flag `eval`/`exec`, bare `except`, and subprocess `shell=True`; not an LLM review or full security audit
- **CLI learning experiment**: `experiments/05_code_review_agent.py` still supports local AST checks and optional Claude API use
- **Groq backend reference**: `experiments/groq_vercel_api_reference.py` is retained for study and tests, not deployed
- **Sample**: `experiments/samples/review_me.py` (intentionally uses `eval`)
- **Notes**: `notes/day5_code_review.md`
- **Tests**: `npm run test:review` and `python -m unittest discover -s tests -v`

### Day 6: Transformer Architecture and Self-Attention
- **Experiment**: `experiments/06_attention_visualization.py` — demonstrates how attention weights connect tokens
- **Concepts**: Self-attention, Q/K/V vectors, multi-head attention, O(n²) complexity
- **Key insight**: Attention is quadratic — doubling context quadruples computation
- **Notes**: `notes/day6_transformers.md`

### Day 7: Complete Transformer Block and Text Generation
- **Experiment**: `experiments/07_transformer_block.py` — shows full transformer block and autoregressive generation
- **Concepts**: Feed-forward networks, layer normalization, residual connections, autoregressive decoding
- **Key insight**: Generation is slow because it's inherently sequential; each token depends on all previous
- **Notes**: `notes/day7_transformer_block.md`

### Day 8: Model Internals - VRAM and Quantization
- **Experiment**: `experiments/08_vram_calculator.py` — calculates VRAM for different models and precisions
- **Concepts**: Parameters, training vs inference, FP32/FP16/INT8/INT4 quantization, KV cache, CPU offloading
- **Key insight**: INT4 quantization lets your 4GB GPU run 7-8B parameter models
- **Notes**: `notes/day8_vram.md`

### Day 9: AI Agents - Structured Outputs and Tool Calling
- **Experiment**: `experiments/09_agent_structured_output.py` — simulates agent loop with tool calling
- **Concepts**: Structured outputs, tool calling, agent loops, planning
- **Key insight**: Agents use tools to take actions — VRAM calculator, token counter, code reviewer
- **Notes**: `notes/day9_agents.md`

### Day 10: LLM APIs - Structured Outputs and Error Handling
- **Experiment**: `experiments/10_api_patterns.py` — demonstrates API call structure, structured JSON outputs, and error handling
- **Concepts**: API authentication, message format, temperature/max_tokens parameters, JSON schemas, retry logic, rate limiting
- **Key insight**: Structured outputs with JSON schemas eliminate parsing errors; exponential backoff prevents crashes on rate limits
- **Notes**: `notes/day10_apis.md`

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-engineering-learning.git
cd ai-engineering-learning

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run experiments
python experiments/01_tokenize_and_count.py
python experiments/02_context_and_counting.py
python experiments/03_caching_impact.py
python experiments/04_model_comparison_demo.py

# Run local static checks (no API key or API calls)
python experiments/05_code_review_agent.py experiments/samples/review_me.py

# Run Python tests (provider API calls are mocked)
python -m unittest discover -s tests -v
```

Run browser-check tests (Node.js, no dependencies):

```bash
npm run test:review
```

### Deploy the static review demo to Vercel

1. Push the project to GitHub and import the repository into Vercel.
2. Set the framework preset to **Other**, leave Build Command and Install Command empty, and use `public` as the Output Directory.
3. Deploy. No `GROQ_API_KEY`, `APP_PASSWORD`, Python function, or paid model call is required by this static page.

The browser checks stay local. Vercel hosting is subject to your account's plan limits and terms, so this is a no-model-cost design, not a guarantee that every hosting account has zero cost. Check Vercel's current plan limits before sharing it publicly.

## Tech Stack

- Python 3.13
- tiktoken (OpenAI tokenizers)
- transformers (Hugging Face tokenizers)

## Learning Approach

1. Learn the concept intuitively, then technically
2. Implement small experiments to verify understanding
3. Document insights and limitations
4. Build progressively toward AI agent systems

## What's Next

- Extend the code reviewer with a small, explicit tool loop and evaluation set
- Future: RAG, tool calling, multi-agent systems

---

*Part of my journey toward AI Engineer / Agent Engineer roles. Follow along as I build foundational skills in LLM systems.*
