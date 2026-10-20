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
```

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

- Day 4: Model comparison framework
- Day 5: Building a simple agent loop
- Future: RAG, tool calling, multi-agent systems

---

*Part of my journey toward AI Engineer / Agent Engineer roles. Follow along as I build foundational skills in LLM systems.*
