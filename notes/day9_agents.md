# Day 9: Building AI Agents - Structured Outputs and Tool Calling

## What is an AI Agent?

An **AI agent** is an LLM that takes **actions** in an environment, not just generates text. The key capabilities:

1. **Structured outputs**: Force the model to return JSON instead of free text
2. **Tool calling**: Let the model invoke functions
3. **Agent loops**: Repeat "plan → act → observe → replan" until a goal is achieved

## Structured Outputs

**Why they matter:** Without structured outputs, you parse free text from an LLM — fragile and error-prone. With structured outputs, you define a JSON schema and enforce it.

**Example schema:**
```python
{
    "task": "string",
    "model_type": "string",
    "parameter_range": "string",
    "quantization": "string",
    "reasoning": "string",
    "confidence": "number"
}
```

The model returns exactly this structure — no parsing needed.

## Tool Calling

Instead of just answering, the model can call functions:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "calculate_vram",
        "description": "Calculate VRAM needed for a model",
        "parameters": {
            "type": "object",
            "properties": {
                "params": {"type": "number"},
                "dtype": {"type": "string"},
                "seq_len": {"type": "number"}
            }
        }
    }
}]
```

When the model decides it needs to call a tool, it returns a `tool_calls` object instead of text.

## Agent Loop

The full loop:

```
1. User: "Check VRAM for 7B model"
2. LLM: "I'll use the calculator" + tool_call(calculate_vram, ...)
3. Tool: Returns {"total_gb": 4.99}
4. LLM: "Needs 4.99GB, won't fit in 4GB. Use Qwen 4B instead."
```

## Your Agent Project Path

**Phase 1 (next):** Build a **Learning Assistant Agent**
- Input: Your learning goals + hardware constraints
- Tools: VRAM calculator, token counter, cost estimator
- Goal: Recommend what you can build/run on your hardware

**Phase 2:** Expand to a **Model Advisor Agent**
- More tools: performance profiler, quantization guide
- Output: Structured comparison + recommendation

**Phase 3:** Final project — **Curriculum Agent**
- Input: Job description
- Tools: Model advisor + learning resources
- Output: Personalized roadmap

## Files Created

- `experiments/09_agent_structured_output.py` - Agent loop simulation
- `notes/day9_agents.md` - This file

Run: `python experiments/09_agent_structured_output.py`
