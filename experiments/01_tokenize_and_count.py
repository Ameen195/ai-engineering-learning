"""
Day 1 Experiment: Tokenization comparison across model families.

We tokenize the same sentences with two different tokenizers:
  1. tiktoken (cl100k_base) — used by GPT-4, GPT-3.5-turbo, and newer OpenAI models
  2. Llama 3 tokenizer (via Hugging Face transformers) — used by Meta's Llama 3 family

Goal: see how the SAME text produces DIFFERENT token counts and different token pieces.
"""

import sys

# --- Sample sentences covering different text types ---
SAMPLES = [
    ("English prose",
     "The quick brown fox jumps over the lazy dog."),
    ("Code snippet",
     "def fibonacci(n):\n    if n <= 1:\n        return n"),
    ("Numbers and symbols",
     "42 + 7 = 49. Price: $12.99 (50% off!)"),
    ("Mixed technical",
     "The API returned HTTP 200 with {\"tokens\": 1542} in 0.3s."),
    ("Non-ASCII",
     "Café résumé naïve — über cool 🎉"),
]


def load_tiktoken():
    """Load the cl100k_base encoding (GPT-4 family)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return enc


def tokenize_with_tiktoken(enc, text):
    """Return (token_count, list_of_token_strings)."""
    token_ids = enc.encode(text)
    token_strings = [enc.decode([tid]) for tid in token_ids]
    return len(token_ids), token_strings


def load_llama_tokenizer():
    """
    Load the Llama 3 tokenizer from Hugging Face.
    This downloads only the tokenizer files (~few MB), NOT model weights.
    Falls back to a generic SentencePiece tokenizer if Llama 3 is unavailable.
    """
    from transformers import AutoTokenizer

    # Try Llama 3 tokenizer (publicly accessible as of 2024)
    try:
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            use_fast=True,
        )
        print("[INFO] Loaded Llama 3 tokenizer (meta-llama/Meta-Llama-3-8B)")
        return tok, "Llama-3"
    except Exception as e:
        print(f"[WARN] Could not load Llama 3 tokenizer: {e}")
        print("[INFO] Falling back to Llama 2 tokenizer (open access)...")

    # Fallback: Llama 2 uses a SentencePiece tokenizer that is publicly available
    try:
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            use_fast=True,
        )
        print("[INFO] Loaded fallback Llama tokenizer (hf-internal-testing)")
        return tok, "Llama-fallback"
    except Exception as e2:
        print(f"[ERROR] Could not load any Llama tokenizer: {e2}")
        return None, None


def tokenize_with_llama(tok, text):
    """Return (token_count, list_of_token_strings)."""
    # Llama tokenizers return token IDs; we decode each one individually
    token_ids = tok.encode(text, add_special_tokens=False)
    token_strings = [tok.decode([tid]) for tid in token_ids]
    return len(token_ids), token_strings


def format_tokens(token_strings, max_show=20):
    """Format token strings for display, quoting each one."""
    if len(token_strings) <= max_show:
        return " | ".join(repr(t) for t in token_strings)
    shown = token_strings[:max_show]
    return " | ".join(repr(t) for t in shown) + f" | ... (+{len(token_strings) - max_show} more)"


def main():
    print("=" * 70)
    print("DAY 1 EXPERIMENT: Tokenization Comparison")
    print("=" * 70)

    # --- Load tokenizers ---
    print("\n[1] Loading tokenizers...\n")
    tiktok_enc = load_tiktoken()
    print(f"  tiktoken (cl100k_base): vocabulary size = {tiktok_enc.n_vocab:,}")

    llama_tok, llama_name = load_llama_tokenizer()
    if llama_tok is None:
        print("\n[ABORT] Cannot proceed without a second tokenizer.")
        sys.exit(1)

    if hasattr(llama_tok, "vocab_size"):
        print(f"  {llama_name} tokenizer: vocabulary size = {llama_tok.vocab_size:,}")
    else:
        print(f"  {llama_name} tokenizer loaded (vocab size not directly exposed)")

    # --- Run comparison ---
    print("\n[2] Tokenizing sample sentences...\n")
    print("-" * 70)

    for label, text in SAMPLES:
        tik_count, tik_tokens = tokenize_with_tiktoken(tiktok_enc, text)
        llama_count, llama_tokens = tokenize_with_llama(llama_tok, text)

        print(f"  Sample: {label}")
        print(f"  Text:   {repr(text)}")
        print(f"  tiktoken (cl100k_base): {tik_count:3d} tokens")
        print(f"    pieces: {format_tokens(tik_tokens)}")
        print(f"  {llama_name:16s}: {llama_count:3d} tokens")
        print(f"    pieces: {format_tokens(llama_tokens)}")

        diff = tik_count - llama_count
        if diff > 0:
            print(f"  --> tiktoken uses {diff} MORE token(s)")
        elif diff < 0:
            print(f"  --> tiktoken uses {abs(diff)} FEWER token(s)")
        else:
            print(f"  --> Same token count!")
        print("-" * 70)

    # --- Summary ---
    print("\n[3] Summary")
    print("  - Same text -> different token counts across tokenizers")
    print("  - Code and symbols often tokenize very differently")
    print("  - Token count affects: context window usage, API cost, latency")
    print("  - You CANNOT compare token counts across models directly")
    print("=" * 70)


if __name__ == "__main__":
    main()
