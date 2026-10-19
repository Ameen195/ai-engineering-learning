# Day 1: Tokens and Tokenization

## What I learned

### What is a token?
- A token is the basic unit a language model processes — an integer ID mapped to a string in a vocabulary.
- Tokens are NOT words. They are subword pieces learned during tokenizer training.
- Example: "tokenization" might be 1 token in one tokenizer and 3-4 in another.

### How text becomes tokens
1. Raw text -> tokenizer splits it into subword pieces
2. Each piece maps to an integer ID via the vocabulary lookup table
3. The model only ever sees the integer sequence, never the raw text

### Why tokenizers differ
- Each model family trains its own vocabulary using BPE (Byte Pair Encoding) or SentencePiece.
- Different training data and vocabulary sizes produce different splits.
- cl100k_base (GPT-4 family): ~100K vocabulary, optimized for English + code.
- Llama tokenizers: different vocabulary, different split decisions.

### Key observations from the experiment
- English prose: tokenizers often agree roughly, but exact counts differ.
- Code: can differ significantly — one tokenizer may merge `def` + ` ` into one token, another keeps them separate.
- Numbers and symbols: highly tokenizer-dependent. "42" might be 1 token or 2.
- Emoji and non-ASCII: often split into multiple tokens, sometimes differently.
- Punctuation: attached to words in one tokenizer, separate in another.

### Why this matters for AI engineering
- **Context window**: measured in tokens. Same prompt = different token counts = different remaining context.
- **API cost**: priced per input token + per output token. Same text costs differently on different models.
- **Coding agents**: each tool call re-sends accumulated context, so token counts grow multiplicatively.
- **You cannot directly compare token counts across models** — they use different tokenizers.

### Limitations of comparing token counts
- Different vocabularies mean 1 token in model A ≠ 1 token in model B in information content.
- A model with fewer tokens for the same text isn't necessarily "better" — its embeddings and attention are tuned for its own tokenizer.
- Cost comparison should use price-per-token, not raw token counts.
- Benchmark results are per-model; you can't swap tokenizers on a trained model.

## Vocabulary
- **BPE (Byte Pair Encoding)**: algorithm that merges frequent byte pairs iteratively to build a vocabulary.
- **SentencePiece**: tokenizer framework that works on raw bytes, used by Llama and others.
- **cl100k_base**: the specific tiktoken encoding used by GPT-4 and GPT-3.5-turbo.
- **Vocabulary size**: number of unique tokens the tokenizer knows (e.g., ~100K for cl100k_base).

## Files created
- `experiments/01_tokenize_and_count.py` — tokenization comparison script
- `requirements.txt` — tiktoken + transformers
