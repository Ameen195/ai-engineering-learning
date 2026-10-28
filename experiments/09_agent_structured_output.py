"""
Day 9: AI Agent - Structured Output and Tool Calling

Demonstrates:
- Structured outputs (JSON schema enforcement)
- Tool calling simulation (without API costs)
- Agent loop: plan → act → observe → respond
"""

import json
from typing import Dict, List, Any, Optional

# --- Simulated LLM Response ---
# In a real agent, this would be an API call.
# For Day 9, we simulate the LLM response to avoid using API credits.

def simulate_llm_with_tools(messages: List[Dict], tools: List[Dict]) -> Dict:
    """
    Simulates what a real LLM would return when given tools.
    Returns either a tool_call or a text response.
    """
    last_message = messages[-1]["content"].lower()

    # The LLM decides: "I need to call the calculate_vram tool"
    if "vram" in last_message and "calculate" not in last_message:
        params_mention = None
        dtype = None
        seq_len = None

        # Parse parameters from the user's question
        if "8b" in last_message or "8billion" in last_message:
            params_mention = 8e9
        elif "7b" in last_message:
            params_mention = 7e9
        elif "4b" in last_message or "4 billion" in last_message:
            params_mention = 4e9
        elif "3.8b" in last_message or "phi" in last_message:
            params_mention = 3.8e9

        if "int4" in last_message:
            dtype = "int4"
        elif "int8" in last_message:
            dtype = "int8"
        elif "fp16" in last_message:
            dtype = "fp16"

        if "4k" in last_message or "4096" in last_message:
            seq_len = 4096
        elif "2k" in last_message or "2048" in last_message:
            seq_len = 2048
        elif "8k" in last_message or "8192" in last_message:
            seq_len = 8192

        if params_mention and dtype:
            # Simulate tool call
            return {
                "tool_calls": [{
                    "function": {
                        "name": "calculate_vram",
                        "arguments": json.dumps({
                            "params": params_mention,
                            "dtype": dtype,
                            "seq_len": seq_len or 2048
                        })
                    }
                }]
            }

    # Otherwise, respond with structured output
    return {
        "tool_calls": None,
        "content": json.dumps({
            "task": "Learning Recommendation",
            "model_type": "local",
            "parameter_range": "3.8B-4B",
            "quantization": "int4",
            "reasoning": "Your GTX 1650 has 4GB VRAM. Based on VRAM calculations, models in the 3.8-4B parameter range with INT4 quantization will fit within your hardware constraints.",
            "tools_used": ["calculate_vram"],
            "confidence": 0.95
        })
    }

# --- Tools ---

# Model data
MODEL_INFO = {
    "qwen_4b": {"params": 4e9, "dtype": "int4", "vram_gb": 2.63},
    "phi3_mini": {"params": 3.8e9, "dtype": "int4", "vram_gb": 2.51},
    "llama3_8b": {"params": 8e9, "dtype": "int4", "vram_gb": 4.99},
    "gemma_2b": {"params": 2e9, "dtype": "int4", "vram_gb": 1.31},
}

# VRAM calculation (from Day 8)
DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5, "int2": 0.25}

def calculate_vram(params: float, dtype: str, seq_len: int = 2048) -> Dict[str, Any]:
    """Calculate VRAM requirements for a given model configuration."""
    bytes_per = DTYPE_BYTES[dtype]
    weights_gb = (params * bytes_per) / (1024 ** 3)
    kv_cache_gb = (2 * 32 * 4096 * seq_len * bytes_per) / (1024 ** 3)
    activations_gb = weights_gb * 0.15
    overhead_gb = (weights_gb + kv_cache_gb + activations_gb) * 0.1
    total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb

    return {
        "weights_gb": round(weights_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "total_gb": round(total_gb, 2),
        "fits_4gb": total_gb <= 4.0
    }

# Tool registry
TOOLS = {
    "calculate_vram": calculate_vram
}

# Output schema for recommendations
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string"},
        "model_type": {"type": "string"},
        "parameter_range": {"type": "string"},
        "quantization": {"type": "string"},
        "reasoning": {"type": "string"},
        "tools_used": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"}
    }
}

def agent_loop(user_input: str, hardware: str = "GTX 1650 4GB") -> str:
    """
    Simulate an agent loop:
    1. LLM decides whether to call a tool or respond directly
    2. If tool call: execute tool, add result to conversation, repeat
    3. If no tool call: return structured response
    """
    messages = [
        {"role": "system", "content": f"You are an AI engineering advisor. Hardware: {hardware}. Recommend models and tools."},
        {"role": "user", "content": user_input}
    ]

    print("=" * 70)
    print("AGENT LOOP DEMONSTRATION")
    print("=" * 70)
    print(f"User: {user_input}")
    print(f"Hardware: {hardware}")
    print("-" * 70)

    # Step 1: LLM responds
    response = simulate_llm_with_tools(messages, [])
    print("\n[LLM Response]")

    if response.get("tool_calls"):
        # Step 2: Tool call
        tool_call = response["tool_calls"][0]
        print(f"  -> Tool call: {tool_call['function']['name']}")
        print(f"     Arguments: {tool_call['function']['arguments']}")

        # Step 3: Execute tool
        args = json.loads(tool_call["function"]["arguments"])
        result = TOOLS[tool_call["function"]["name"]](**args)
        print(f"  -> Tool result: {result}")

        # Step 4: Add result to conversation, LLM generates final response
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        messages.append({"role": "tool", "content": json.dumps(result)})

        final_response = simulate_llm_with_tools(messages, [])

        if final_response.get("content"):
            print(f"\n[Final LLM Response]")
            recommendation = json.loads(final_response["content"])
            for key, value in recommendation.items():
                print(f"  {key}: {value}")
        else:
            print("  (No final response generated)")
    else:
        # Direct structured response
        if response.get("content"):
            recommendation = json.loads(response["content"])
            print(f"\n[Structured Recommendation]")
            for key, value in recommendation.items():
                print(f"  {key}: {value}")

    print("-" * 70)
    return "Agent loop completed"

def demonstrate_structured_output():
    """Show how structured outputs work without parsing free text."""

    print("=" * 70)
    print("DAY 9: AI AGENTS - STRUCTURED OUTPUTS AND TOOL CALLING")
    print("=" * 70)

    # Part 1: Structured output demonstration
    print("\n1. STRUCTURED OUTPUT DEMO")
    print("-" * 70)
    print("Without structured output, you parse free text (fragile):")
    print("  LLM says: 'I think you should use Qwen 4B with INT4...' ")
    print("  You parse: extract 'Qwen 4B', 'INT4', etc. from text.")
    print()
    print("With structured output, LLM returns JSON (reliable):")

    # Simulate structured response
    messages = [
        {"role": "system", "content": "Return your recommendation as strict JSON matching the schema."},
        {"role": "user", "content": "What model should I run on a GTX 1650 with 4GB VRAM?"}
    ]

    response = simulate_llm_with_tools(messages, [])
    if response.get("content"):
        rec = json.loads(response["content"])
        print(json.dumps(rec, indent=2))

    # Part 2: Tool calling demonstration
    print("\n\n2. TOOL CALLING DEMO")
    print("-" * 70)

    # Simulate agent with VRAM calculation tool
    user_query = "I want to run a 7B parameter model on my GTX 1650 4GB. Can you check VRAM?"
    messages = [
        {"role": "system", "content": "You have access to a VRAM calculation tool. Use it when asked about memory requirements."},
        {"role": "user", "content": user_query}
    ]

    response = simulate_llm_with_tools(messages, [])
    print(f"User: {user_query}")
    print(f"LLM decision: Call calculate_vram tool")
    print(f"Tool arguments: params=7e9, dtype=int4, seq_len=2048")

    result = calculate_vram(7e9, "int4", 2048)
    print(f"Tool result: {result}")
    print(f"\nLLM final response: 'The 7B model needs {result['total_gb']} GB VRAM in INT4. Your GTX 1650 has 4GB, so it won't fit. Use a 4B model like Qwen or Phi-3 Mini instead.'")

    # Part 3: Full agent loop
    print("\n\n3. FULL AGENT LOOP DEMO")
    print("-" * 70)

    # Scenario 1: VRAM check
    print("\n--- Scenario 1: VRAM Check ---")
    agent_loop("I want to run a 7B parameter model on my GTX 1650 4GB. Can you check VRAM?")

    print("\n\n--- Scenario 2: Model Recommendation ---")
    # Scenario 2: Direct recommendation (LLM knows the constraints)
    messages2 = [
        {"role": "system", "content": "You are an AI engineering advisor. Hardware: GTX 1650 4GB. Recommend models."},
        {"role": "user", "content": "What local models can I run?"}
    ]
    response2 = simulate_llm_with_tools(messages2, [])
    if response2.get("content"):
        rec2 = json.loads(response2["content"])
        print(json.dumps(rec2, indent=2))

    # Part 4: Available tools for your project
    print("\n\n4. TOOLS FOR YOUR LEARNING ASSISTANT AGENT")
    print("-" * 70)
    print("Your agent can use these tools (you've built them all in 9 days):")
    print("  1. calculate_vram      (Day 8) - VRAM requirements")
    print("  2. count_tokens        (Day 1) - Token counting")
    print("  3. compare_models      (Day 4) - Cost/latency comparison")
    print("  4. review_code         (Day 5) - Pattern checking")
    print("  5. explain_transformer (Day 7) - Architecture explanations")

    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Structured outputs eliminate parsing errors - JSON in, JSON out")
    print("2. Tool calling lets LLMs take actions - not just answer questions")
    print("3. Agent loop: LLM decides to act → tool executes → LLM responds")
    print("4. You can build agents incrementally - each day's learning is a tool")
    print("5. No API key needed for simulation - but production needs real LLM calls")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_structured_output()
