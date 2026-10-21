"""
Day 4 Experiment: Model Comparison Framework (Demo Version)

This version uses mock data to demonstrate the framework structure.
The real version would call actual APIs (Ollama, OpenAI, Anthropic).

We'll compare models on identical tasks, measuring:
- Latency (time to first token, tokens per second)
- Token usage (input/output counts)
- Cost (for cloud models)
"""

import tiktoken
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ModelResult:
    """Store results from a single model run."""
    model_name: str
    task: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    time_to_first_token: float  # seconds
    tokens_per_second: float
    total_latency: float  # seconds
    cost_usd: float
    success: bool


# --- Test tasks ---
TASKS = [
    {
        "name": "coding_basic",
        "prompt": "Write a Python function to check if a number is prime. Include docstring.",
        "category": "coding",
    },
    {
        "name": "reasoning_math",
        "prompt": "If a train travels at 60 mph for 2.5 hours, then 80 mph for 1.5 hours, what is the total distance? Show your work.",
        "category": "reasoning",
    },
    {
        "name": "explanation",
        "prompt": "Explain what a context window is in LLMs in 2-3 sentences.",
        "category": "knowledge",
    },
    {
        "name": "summarization",
        "prompt": "Summarize this in one sentence: 'Python is a high-level programming language known for its readability. It supports multiple paradigms including OOP and functional programming. Python has a large standard library and active community.'",
        "category": "summarization",
    },
]


def count_tokens(text: str, model: str = "local") -> int:
    """Count tokens using appropriate tokenizer."""
    if model == "local":
        enc = tiktoken.get_encoding("cl100k_base")
    else:
        enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def mock_local_model(model_name: str, tasks: List[Dict]) -> List[ModelResult]:
    """
    Generate mock results for local model.
    In production, replace with actual Ollama API calls.
    """
    results = []
    
    print(f"\nTesting local model: {model_name} (mock data)")
    print("-" * 70)
    
    for task in tasks:
        input_tokens = count_tokens(task["prompt"], model="local")
        output_tokens = int(len(task["prompt"]) * 2)  # Rough estimate
        
        # Local model characteristics: slower, free
        ttft = 0.8  # 800ms to first token
        tps = 20.0  # 20 tokens/sec generation speed
        total_time = ttft + (output_tokens / tps)
        
        result = ModelResult(
            model_name=model_name,
            task=task["name"],
            prompt=task["prompt"],
            response="[Mock response - replace with Ollama API call]",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_to_first_token=ttft,
            tokens_per_second=tps,
            total_latency=total_time,
            cost_usd=0.0,  # Local = free
            success=True,
        )
        
        print(f"\nTask: {task['name']}")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  TTFT: {ttft:.2f}s, TPS: {tps:.1f}, Total: {total_time:.2f}s")
        print(f"  Cost: $0.000000")
        
        results.append(result)
    
    return results


def mock_cloud_model(model_name: str, tasks: List[Dict], pricing: Dict) -> List[ModelResult]:
    """
    Generate mock results for cloud models.
    In production, replace with actual API calls (OpenAI, Anthropic, etc.)
    """
    results = []
    
    print(f"\nTesting cloud model: {model_name} (mock data)")
    print("-" * 70)
    
    for task in tasks:
        # Cloud models often use fewer tokens (better tokenization)
        input_tokens = int(count_tokens(task["prompt"], model="local") * 0.85)
        output_tokens = int(len(task["prompt"]) * 1.5)
        
        # Cloud models are faster
        ttft = 0.3  # 300ms
        tps = 50.0  # 50 tokens/sec
        total_time = ttft + (output_tokens / tps)
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing["input_price"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_price"]
        cost = input_cost + output_cost
        
        result = ModelResult(
            model_name=model_name,
            task=task["name"],
            prompt=task["prompt"],
            response="[Mock response - replace with API call]",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_to_first_token=ttft,
            tokens_per_second=tps,
            total_latency=total_time,
            cost_usd=cost,
            success=True,
        )
        
        print(f"\nTask: {task['name']}")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  TTFT: {ttft:.2f}s, TPS: {tps:.1f}, Total: {total_time:.2f}s")
        print(f"  Cost: ${cost:.6f}")
        
        results.append(result)
    
    return results


def compare_results(all_results: List[ModelResult]):
    """Generate comparison report."""
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)
    
    # Group by model
    models = {}
    for result in all_results:
        if result.model_name not in models:
            models[result.model_name] = []
        models[result.model_name].append(result)
    
    # Summary table
    print(f"\n{'Model':<25} {'Avg TTFT':<12} {'Avg TPS':<12} {'Total Cost':<12} {'Tasks':<8}")
    print("-" * 70)
    
    for model_name, results in models.items():
        avg_ttft = sum(r.time_to_first_token for r in results) / len(results)
        avg_tps = sum(r.tokens_per_second for r in results) / len(results)
        total_cost = sum(r.cost_usd for r in results)
        task_count = len(results)
        
        print(f"{model_name:<25} {avg_ttft:<12.2f} {avg_tps:<12.1f} ${total_cost:<11.6f} {task_count:<8}")
    
    # Per-task comparison
    print("\n\nPer-Task Breakdown:")
    print("-" * 70)
    
    tasks = set(r.task for r in all_results)
    for task in sorted(tasks):
        print(f"\nTask: {task}")
        print(f"  {'Model':<25} {'Input':<8} {'Output':<8} {'TTFT':<8} {'TPS':<8} {'Cost':<10}")
        
        task_results = [r for r in all_results if r.task == task]
        for r in task_results:
            print(f"  {r.model_name:<25} {r.input_tokens:<8} {r.output_tokens:<8} {r.time_to_first_token:<8.2f} {r.tokens_per_second:<8.1f} ${r.cost_usd:<9.6f}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    local_results = [r for r in all_results if "local" in r.model_name.lower() or "qwen" in r.model_name.lower()]
    cloud_results = [r for r in all_results if "local" not in r.model_name.lower() and "qwen" not in r.model_name.lower()]
    
    if local_results and cloud_results:
        local_cost = sum(r.cost_usd for r in local_results)
        cloud_cost = sum(r.cost_usd for r in cloud_results)
        
        local_tps = sum(r.tokens_per_second for r in local_results) / len(local_results)
        cloud_tps = sum(r.tokens_per_second for r in cloud_results) / len(cloud_results)
        
        print(f"\nLocal model: ${local_cost:.6f} total, {local_tps:.1f} avg TPS")
        print(f"Cloud models: ${cloud_cost:.6f} total, {cloud_tps:.1f} avg TPS")
        print(f"\nTradeoff: Local is free but {cloud_tps/local_tps:.1f}x slower")
        print(f"Cloud is {cloud_tps/local_tps:.1f}x faster but costs ${cloud_cost:.6f}")


def main():
    print("=" * 70)
    print("DAY 4 EXPERIMENT: Model Comparison Framework (Demo)")
    print("=" * 70)
    print("\nNote: This uses mock data to demonstrate the framework.")
    print("Replace mock functions with real API calls for production use.")
    
    all_results = []
    
    # Test 1: Local model (mock)
    local_model = "qwen3.5:4b (local)"
    local_results = mock_local_model(local_model, TASKS)
    all_results.extend(local_results)
    
    # Test 2: GPT-4 (mock)
    gpt4_pricing = {"input_price": 10.0, "output_price": 30.0}
    gpt4_results = mock_cloud_model("gpt-4-turbo", TASKS, gpt4_pricing)
    all_results.extend(gpt4_results)
    
    # Test 3: Claude 3.5 (mock)
    claude_pricing = {"input_price": 3.0, "output_price": 15.0}
    claude_results = mock_cloud_model("claude-3.5-sonnet", TASKS, claude_pricing)
    all_results.extend(claude_results)
    
    # Generate comparison
    compare_results(all_results)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("1. Local models: Free but slower, limited by your hardware")
    print("2. Cloud models: Fast and powerful, but cost per token")
    print("3. Choose based on: task complexity, latency needs, budget")
    print("4. For production: Test on real tasks, measure quality + cost")
    print("5. Framework ready - add real API calls to extend it")
    print("=" * 70)


if __name__ == "__main__":
    main()
