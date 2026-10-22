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

### Day 5: Python Code Review Agent
- **CLI**: `experiments/05_code_review_agent.py` runs local AST checks and supports optional Claude review
- **Web app**: `public/index.html` with a private Vercel Python endpoint at `api/review.py` using Groq
- **Local checks**: AST rules for `eval`/`exec`, bare `except`, `shell=True`, and syntax errors
- **Scope**: One read-only Python file up to 25 KB; code is not executed or stored
- **Sample**: `experiments/samples/review_me.py` (intentionally uses `eval`)
- **Notes**: `notes/day5_code_review.md`
- **Offline tests**: `python -m unittest discover -s tests -v`

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

# Run offline tests (no keys or API calls needed)
python -m unittest discover -s tests -v
```

For optional Claude review, set `ANTHROPIC_API_KEY` and pass `--claude` as documented in `notes/day5_code_review.md`.

### Deploy the private Groq web demo to Vercel

1. Push the project to GitHub and import the repository into Vercel.
2. In Vercel Project Settings → Environment Variables, add `GROQ_API_KEY`, a unique `APP_PASSWORD` of at least 20 characters, and optionally `GROQ_MODEL` (default: `openai/gpt-oss-20b`). Apply secrets to Production and Preview only as needed.
3. Enable Vercel Deployment Protection for the deployment before using it; the app password is an additional gate, not account-level authentication.
4. Deploy, then open the protected deployment URL. Enter the `APP_PASSWORD` value in the page to review a sample `.py` file.

Groq usage and free-tier eligibility depend on current model/account limits; check the Groq Console before use. Never expose or commit API keys. The web app sends source to Groq and does not persist it.

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
