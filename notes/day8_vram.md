# Day 8: Model Internals - Parameters, Quantization, and VRAM

## Parameters

**What they are:** The learned weights inside a model. Each parameter is a single floating-point number that the model adjusts during training.

**How many:**
- GPT-2 XL: 1.5 billion parameters
- GPT-3: 175 billion parameters
- Llama 3 8B: 8 billion parameters
- Llama 3 70B: 70 billion parameters

**Memory for weights alone:**
- FP32 (4 bytes): 1B params = 4 GB
- FP16 (2 bytes): 1B params = 2 GB
- INT4 (0.5 bytes): 1B params = 0.5 GB

## Training vs. Inference

**Training:**
- Adjusts ALL parameters simultaneously
- Needs to store gradients (same size as parameters)
- Needs optimizer state (Adam: 2x parameter size)
- Memory = Parameters + Gradients + Optimizer + Activations
- Result: 8-16x more memory than inference

**Inference:**
- Uses frozen parameters
- Only needs parameters + activations + KV cache
- Much smaller memory footprint
- Can be heavily quantized

## Quantization

Quantization reduces the precision of model weights to save memory and improve speed:

| Format | Bytes | Memory Savings | Quality Loss |
|--------|-------|---------------|--------------|
| FP32   | 4     | 1x (baseline) | None |
| FP16/BF16 | 2   | 2x | Negligible |
| INT8   | 1     | 4x | Minor |
| INT4   | 0.5   | 8x | Noticeable |
| INT2   | 0.25  | 16x | Significant |

**When to use each:**
- **FP16**: Cloud inference, high quality required
- **INT8**: Good balance for most applications
- **INT4**: Local inference on consumer GPUs (like yours)
- **INT4 with Q4_K_M**: Best quality 4-bit quantization (used by Ollama, llama.cpp)

## VRAM Calculation

Total VRAM needed for inference:

```
VRAM = Model_weights + KV_cache + Activations + Overhead(10%)

KV_cache = 2 × num_layers × hidden_size × seq_len × bytes_per_param
```

### Example: 7B model in INT4 (0.5 bytes/param)

- **Model weights:** 7B × 0.5 = 3.5 GB
- **KV cache (2K context):** ~0.15 GB
- **Activations:** ~0.5 GB
- **Overhead:** ~0.4 GB
- **Total:** ~4.5 GB

This is why INT4 lets you run 7B-8B models on your GTX 1650 (4GB)!

## KV Cache

During text generation, the model caches attention **keys** and **values** for previously processed tokens. This avoids recomputing attention for tokens that haven't changed.

**Impact:**
- KV cache grows linearly with sequence length
- At 128K context: KV cache can be 5-10 GB (larger than model weights!)
- This is a major bottleneck for long-context inference

## CPU Offloading

When GPU VRAM is insufficient:
- Move some model layers to CPU memory
- Trade speed for the ability to run larger models
- Tools: `llama.cpp --gpu-layers`, `Ollama --gpu-layers`, `vLLM CPU offloading`

## Training Memory Breakdown

Training needs drastically more memory:
- **Parameters:** 7B × 2 bytes = 14 GB
- **Gradients:** Same size = 14 GB
- **Optimizer state (Adam):** ~2x parameters = 28 GB
- **Activations:** ~1x parameters = 14 GB
- **Total:** ~80 GB for a 70B model in FP16

This is why training requires expensive GPUs (A100 80GB, H100 80GB).

## Files Created

- `experiments/08_vram_calculator.py` - VRAM calculation and comparison
- `notes/day8_vram.md` - this file

Run the experiment: `python experiments/08_vram_calculator.py`
