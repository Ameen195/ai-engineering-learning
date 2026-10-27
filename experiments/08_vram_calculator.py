"""
Day 8: VRAM Calculator for LLM Inference

Demonstrates:
- Memory calculation for different model sizes and precisions
- How quantization reduces VRAM requirements
- KV cache impact on context length
- CPU offloading tradeoffs
"""

# Model parameter counts (common models)
MODEL_SIZES = {
    "llama3_8b": 8e9,      # 8B parameters
    "llama3_70b": 70e9,    # 70B parameters
    "gpt2_xl": 1.5e9,      # 1.5B parameters
    "gpt3_175b": 175e9,    # 175B parameters
    "phi3_mini": 3.8e9,    # 3.8B parameters
    "qwen_4b": 4e9,        # 4B parameters
}

# Bytes per parameter for different dtypes
DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "int2": 0.25,
}

def calculate_vram(params, dtype, seq_len=2048, num_layers=32, hidden_size=4096):
    """
    Calculate approximate VRAM needed for inference.

    Breakdown:
    1. Model weights: params * bytes_per_param
    2. KV cache: 2 * num_layers * hidden_size * seq_len * bytes_per_param
       (2 because we store both keys and values)
    3. Activations: Roughly proportional to model size
    4. Overhead: ~10-20% for framework, buffers, etc.
    """
    bytes_per = DTYPE_BYTES[dtype]

    # 1. Model weights
    weights_gb = (params * bytes_per) / (1024 ** 3)

    # 2. KV cache (stores attention keys and values for each layer)
    kv_cache_gb = (2 * num_layers * hidden_size * seq_len * bytes_per) / (1024 ** 3)

    # 3. Activations (rough estimate: ~15-25% of model size per layer)
    activations_gb = weights_gb * 0.15

    # 4. Framework overhead
    overhead_gb = (weights_gb + kv_cache_gb + activations_gb) * 0.1

    total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb

    return {
        "weights_gb": weights_gb,
        "kv_cache_gb": kv_cache_gb,
        "activations_gb": activations_gb,
        "overhead_gb": overhead_gb,
        "total_gb": total_gb,
    }

def demonstrate_vram_calculations():
    """Show how model size, precision, and context affect VRAM."""

    print("=" * 80)
    print("DAY 8: VRAM CALCULATOR FOR LLM INFERENCE")
    print("=" * 80)

    # Part 1: Same model, different precisions
    print("\n1. EFFECT OF QUANTIZATION ON A 7B PARAMETER MODEL")
    print("-" * 60)
    model = "qwen_4b"
    params = MODEL_SIZES[model]

    print(f"Model: Qwen 4B ({params/1e9:.1f}B parameters)")
    print(f"{'Precision':<10} {'Weights':>10} {'KV Cache':>10} {'Total':>10} {'Fits 4GB?':>10}")
    print("-" * 60)

    for dtype in ["fp32", "fp16", "int8", "int4"]:
        result = calculate_vram(params, dtype)
        fits = "Yes" if result["total_gb"] <= 4.0 else "No"
        print(f"{dtype:<10} {result['weights_gb']:>8.2f} GB {result['kv_cache_gb']:>8.2f} GB {result['total_gb']:>8.2f} GB {fits:>10}")

    # Part 2: Different model sizes, same precision
    print("\n2. DIFFERENT MODEL SIZES (INT4 quantization)")
    print("-" * 60)
    print(f"{'Model':<20} {'Params':>12} {'Weights':>10} {'Total':>10} {'Fits 4GB?':>10}")
    print("-" * 60)

    for name, params in MODEL_SIZES.items():
        result = calculate_vram(params, "int4")
        fits = "Yes" if result["total_gb"] <= 4.0 else "No"
        print(f"{name:<20} {params/1e9:>10.1f}B {result['weights_gb']:>8.2f} GB {result['total_gb']:>8.2f} GB {fits:>10}")

    # Part 3: KV cache impact
    print("\n3. KV CACHE IMPACT (7B model, INT4, varying context)")
    print("-" * 60)
    params = MODEL_SIZES["llama3_8b"]
    print(f"Model: Llama 3 8B (INT4)")
    print(f"{'Context':<12} {'Weights':>10} {'KV Cache':>10} {'Total':>10}")
    print("-" * 60)

    for seq_len in [512, 2048, 4096, 8192, 16384, 128000]:
        result = calculate_vram(params, "int4", seq_len=seq_len)
        label = f"{seq_len//1024}K" if seq_len >= 1024 else f"{seq_len}"
        print(f"{label:<12} {result['weights_gb']:>8.2f} GB {result['kv_cache_gb']:>8.2f} GB {result['total_gb']:>8.2f} GB")

    # Part 4: Your hardware
    print("\n4. YOUR HARDWARE: GTX 1650 (4GB VRAM)")
    print("-" * 60)
    print("Models that fit in 4GB VRAM with INT4:")
    print("-" * 60)

    for name, params in MODEL_SIZES.items():
        result = calculate_vram(params, "int4", seq_len=2048)
        if result["total_gb"] <= 4.0:
            print(f"  {name:<20} {result['total_gb']:>6.2f} GB total")

    print("\nWhat to try if you're out of VRAM:")
    print("  - Reduce context length (fewer tokens to cache)")
    print("  - Use INT4 instead of INT8")
    print("  - Offload layers to CPU (llama.cpp, ollama with --gpu-layers)")
    print("  - Use a smaller model (Qwen 0.6B, Phi-3 mini, Gemma 2B)")

    # Part 5: Training vs Inference memory
    print("\n5. TRAINING vs INFERENCE MEMORY")
    print("-" * 60)
    print("Training requires MUCH more memory than inference:")
    print(f"  Weights:         {params/1e9:.1f}B x 2 bytes (fp16) = {(params*2)/(1024**3):.1f} GB")
    print(f"  Gradients:       Same as weights = {(params*2)/(1024**3):.1f} GB")
    print(f"  Optimizer state: ~2x weights (Adam) = {(params*4)/(1024**3):.1f} GB")
    print(f"  Activations:     ~1x weights = {(params*2)/(1024**3):.1f} GB")
    print(f"  Total training:  ~{(params*2*4)/(1024**3):.1f} GB")
    print(f"  (This is why training needs 80+ GB GPUs!)")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("1. Quantization is essential for local inference on consumer GPUs")
    print("2. KV cache grows with context length - 128K context = 100x the KV cache!")
    print("3. Training needs 8-16x more memory than inference (gradients + optimizer)")
    print("4. Your GTX 1650 (4GB) can run 3-8B parameter models with INT4")
    print("5. CPU offloading lets you run larger models at the cost of speed")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_vram_calculations()
