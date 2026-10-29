"""
Day 11: Learning Assistant Agent - Complete Agent Implementation

Integrates tools from Days 1-10:
- Token counting (Day 1)
- Context management (Day 2)  
- Prompt caching simulation (Day 3)
- Model comparison (Day 4)
- Code review patterns (Day 5)
- Transformer concepts (Days 6-7)
- VRAM calculation (Day 8)
- API patterns (Day 10)

Demonstrates:
- Agent loop architecture (plan -> act -> observe -> respond)
- Tool calling integration
- Structured outputs (JSON schemas)
- Cost and token tracking
- Error handling
- Context management
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# --- Hardware Profile ---
HARDWARE_PROFILE = {
    "gpu": "NVIDIA GTX 1650",
    "vram_gb": 4.0,
    "ram_gb": 16,
    "cpu": "Intel Core i7-10870H",
    "os": "Windows 11"
}

# --- JSON Schema for Agent Response ---
AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {
            "type": "string",
            "description": "Type of query: 'learning', 'setup', 'optimization', 'general'"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "How confident the agent is in its response"
        },
        "tools_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tools invoked"
        },
        "token_estimate": {
            "type": "object",
            "properties": {
                "input_tokens": {"type": "integer"},
                "output_tokens": {"type": "integer"},
                "estimated_cost": {"type": "number"}
            }
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "estimated_hours": {"type": "integer"}
                }}
        },
        "answer": {"type": "string"},
        "limitations": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["query_type", "confidence", "tools_used", "token_estimate", "recommendations", "answer", "limitations"]
}

# --- Tools (Integrated from Days 1-10) ---

# VRAM Calculation (from Day 8)
DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5, "int2": 0.25}

def calculate_vram(params: float, dtype: str, seq_len: int = 2048) -> Dict[str, Any]:
    """Calculate VRAM needed for a model."""
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
        "fits_4gb": total_gb <= 4.0,
        "fits_16gb": total_gb <= 16.0
    }

# Token Estimation (from Day 1)
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Rough token estimate without exact tokenizer."""
    if model.startswith("gpt") or model.startswith("claude") or model.startswith("gemini"):
        return len(text) // 3
    return len(text) // 4

# Model Comparison (from Day 4)
MODELS = {
    "qwen3.5:4b": {"provider": "local", "context": 4096, "vram_need": 3.3, "speed_tps": 15},
    "qwen3.5:0.6b": {"provider": "local", "context": 4096, "vram_need": 0.9, "speed_tps": 45},
    "phi3:mini": {"provider": "local", "context": 4096, "vram_need": 1.6, "speed_tps": 28},
    "gpt-4o": {"provider": "cloud", "context": 128000, "vram_need": 0, "speed_tps": 50},
    "claude-3-5-sonnet": {"provider": "cloud", "context": 200000, "vram_need": 0, "speed_tps": 50},
    "gemini-1.5-flash": {"provider": "cloud", "context": 1000000, "vram_need": 0, "speed_tps": 60}
}

def find_compatible_models(hardware_vram: float, min_speed: int = 0) -> List[Dict]:
    """Find models that fit your hardware."""
    compatible = []
    for name, specs in MODELS.items():
        if specs["provider"] == "local":
            if specs["vram_need"] <= hardware_vram and specs["speed_tps"] >= min_speed:
                compatible.append({"name": name, **specs, "estimated_cost": "$0.00 (local)"})
        else:
            compatible.append({"name": name, **specs, "estimated_cost": "$0.001-0.01 per 1K tokens"})
    return compatible

# --- LLM Client Simulation (from Day 10) ---

class LLMClient:
    """Simulated LLM client with error handling and cost tracking."""

    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0

    def call(self, messages: List[Dict], schema: Dict = None,
             temperature: float = 0.7, max_tokens: int = 1000) -> Dict:
        """Simulate an LLM API call with structured response."""
        # In production: return actual API response
        # For this experiment: simulate based on user query

        content = messages[-1]["content"].lower()

        # Token estimation
        input_tokens = estimate_tokens(str(messages))
        output_tokens = min(max_tokens // 10, 100)  # Simulated
        cost = (input_tokens / 1000) * 0.003 + (output_tokens / 1000) * 0.006
        self.total_cost += cost
        self.total_tokens += input_tokens + output_tokens

        # Generate response based on query type
        if "vram" in content or "4gb" in content or "memory" in content:
            return self._vram_response(messages, input_tokens, output_tokens, cost)
        elif "model" in content and ("which" in content or "recommend" in content):
            return self._model_response(messages, input_tokens, output_tokens, cost)
        elif "token" in content:
            return self._token_response(messages, input_tokens, output_tokens, cost)
        else:
            return self._general_response(messages, input_tokens, output_tokens, cost)

    def _vram_response(self, messages, input_tokens, output_tokens, cost):
        compatible = find_compatible_models(HARDWARE_PROFILE["vram_gb"])
        local_models = [m for m in compatible if m["provider"] == "local"]

        result = {
            "query_type": "optimization",
            "confidence": 0.95,
            "tools_used": ["calculate_vram", "find_compatible_models"],
            "token_estimate": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": round(cost, 6)
            },
            "recommendations": [
                {
                    "title": "Use INT4 quantization",
                    "description": f"Your GTX 1650 (4GB VRAM) can run models up to ~4B parameters with INT4 quantization.",
                    "priority": "high",
                    "estimated_hours": 2
                },
                {
                    "title": f"Compatible local models: {[m['name'] for m in local_models]}",
                    "description": "These models fit your hardware with proper quantization.",
                    "priority": "high",
                    "estimated_hours": 3
                }
            ],
            "answer": "With 4GB VRAM, use INT4 quantization. Models like Qwen3.5 4B, Phi-3 Mini, or Qwen3.5 0.6B will work.",
            "limitations": [
                "VRAM estimates are approximate",
                "Actual performance depends on context length and system load",
                "KV cache grows with conversation length"
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    def _model_response(self, messages, input_tokens, output_tokens, cost):
        compatible = find_compatible_models(HARDWARE_PROFILE["vram_gb"], min_speed=15)

        result = {
            "query_type": "setup",
            "confidence": 0.90,
            "tools_used": ["find_compatible_models"],
            "token_estimate": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": round(cost, 6)
            },
            "recommendations": [
                {
                    "title": "Start with local model",
                    "description": "Use Qwen3.5 4B or Phi-3 Mini with Ollama for free development.",
                    "priority": "high",
                    "estimated_hours": 1
                },
                {
                    "title": "Use cloud for production",
                    "description": "Migrate to GPT-4o or Claude 3.5 Sonnet when quality matters more than cost.",
                    "priority": "medium",
                    "estimated_hours": 4
                }
            ],
            "answer": f"For your GTX 1650: {len(compatible)} compatible models found. Local models: free. Cloud models: pay-per-use.",
            "limitations": [
                "Local model quality may be lower than cloud models",
                "Cloud model costs accumulate with usage"
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    def _token_response(self, messages, input_tokens, output_tokens, cost):
        result = {
            "query_type": "learning",
            "confidence": 0.92,
            "tools_used": ["estimate_tokens"],
            "token_estimate": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": round(cost, 6)
            },
            "recommendations": [
                {
                    "title": "Understand tokenization",
                    "description": "Tokens are ~4 chars of text for English. Code may use more tokens.",
                    "priority": "high",
                    "estimated_hours": 2
                }
            ],
            "answer": "Rough rule: 1 token is about 4 characters for English text, 1 token is about 0.75 words.",
            "limitations": [
                "Token estimates vary by tokenizer",
                "Non-English text and code have different ratios"
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    def _general_response(self, messages, input_tokens, output_tokens, cost):
        result = {
            "query_type": "general",
            "confidence": 0.80,
            "tools_used": [],
            "token_estimate": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": round(cost, 6)
            },
            "recommendations": [
                {
                    "title": "Be specific about your question",
                    "description": "Ask about VRAM, models, tokens, or code review for better recommendations.",
                    "priority": "high",
                    "estimated_hours": 0
                }
            ],
            "answer": "Ask me about VRAM, model selection, tokenization, or code review for personalized recommendations.",
            "limitations": [
                "I'm a simulated agent for learning purposes",
                "For production, connect to a real LLM API"
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

# --- Agent Class ---

class LearningAssistantAgent:
    """
    A complete AI agent that helps you learn AI engineering.

    Integrates tools from all previous days:
    - Token estimation (Day 1)
    - Context management (Day 2)
    - Model comparison (Day 4)
    - Code review patterns (Day 5)
    - VRAM calculation (Day 8)
    - API patterns (Day 10)
    """

    def __init__(self, hardware_profile: Dict):
        self.hardware = hardware_profile
        self.client = LLMClient()
        self.conversation_history: List[Dict] = []
        self.max_context_tokens = 4000  # Conservative for local models
        self.cache: Dict[str, str] = {}  # Simple prompt cache (Day 3)

    def run(self, user_query: str) -> Dict:
        """
        Run the complete agent loop:
        1. Parse query
        2. Use relevant tools
        3. Generate structured response
        4. Track costs
        """
        start_time = time.time()

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        self._manage_context()

        # Check cache (Day 3 concept)
        cache_key = self._generate_cache_key(user_query)
        if cache_key in self.cache:
            result = json.loads(self.cache[cache_key])
            result["token_estimate"]["estimated_cost"] = 0  # Cache hit is free
            result["limitations"].append("Response served from cache")
            return result

        # Prepare system message
        system_prompt = self._build_system_prompt()

        # Call LLM with tools
        messages = [system_prompt] + self.conversation_history
        response = self.client.call(
            messages,
            schema=AGENT_RESPONSE_SCHEMA,
            temperature=0.3,  # Deterministic for technical responses
            max_tokens=1500
        )

        # Parse response
        content = response["choices"][0]["message"]["content"]

        # Validate against schema
        try:
            result = json.loads(content)
            # Add metadata
            result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
            result["total_cost_so_far"] = round(self.client.total_cost, 6)
            result["total_tokens_so_far"] = self.client.total_tokens

            # Cache the response
            self.cache[cache_key] = content

            # Add to history
            self.conversation_history.append({"role": "assistant", "content": content})

            return result
        except json.JSONDecodeError as e:
            return {
                "query_type": "error",
                "confidence": 0.0,
                "tools_used": [],
                "token_estimate": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0},
                "recommendations": [],
                "answer": f"Error parsing response: {str(e)}",
                "limitations": ["LLM response was not valid JSON"],
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }

    def _build_system_prompt(self) -> Dict:
        """Build system prompt with hardware profile and tool descriptions."""
        tools_desc = [
            "calculate_vram(params, dtype, seq_len) - Calculate VRAM needed",
            "estimate_tokens(text, model) - Estimate token count",
            "find_compatible_models(vram_gb, min_speed) - Find runnable models",
            "calculate_vram(7e9, 'int4', 2048) - Example: 7B model in INT4"
        ]

        return {
            "role": "system",
            "content": f"""You are an AI Engineering Learning Assistant.

Hardware profile: {json.dumps(self.hardware)}
Available tools: {', '.join(tools_desc)}

Your job: Answer questions about AI engineering, model selection, VRAM, and learning paths.
Use the tools available to provide data-driven recommendations.
Always return structured JSON matching the provided schema.
Keep responses concise and actionable.
"""
        }

    def _manage_context(self):
        """Keep conversation within token budget (Day 2 concept)."""
        # Rough check: if conversation > 3000 tokens, trim oldest
        total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
        if total_chars > 12000:  # ~3000 tokens
            # Remove oldest message pair
            self.conversation_history = self.conversation_history[2:]

    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for prompt caching (Day 3 concept)."""
        # Simple hash - in production use proper hashing
        return str(hash(query[:100]))

    def get_stats(self) -> Dict:
        """Return agent session statistics."""
        return {
            "conversation_turns": len(self.conversation_history) // 2,
            "total_cost": round(self.client.total_cost, 6),
            "total_tokens": self.client.total_tokens,
            "cache_hits": len(self.cache),
            "current_context_length": sum(len(msg["content"]) for msg in self.conversation_history)
        }


# --- Demo ---

def demonstrate_agent():
    """Run the complete agent with multiple queries."""
    print("=" * 70)
    print("DAY 11: LEARNING ASSISTANT AGENT")
    print("=" * 70)

    print("\nHardware Profile:")
    for key, value in HARDWARE_PROFILE.items():
        print(f"  {key}: {value}")

    # Create agent
    agent = LearningAssistantAgent(HARDWARE_PROFILE)

    # Test queries
    queries = [
        "Which models can I run locally on my GTX 1650?",
        "Calculate VRAM for a 7B parameter model in INT4",
        "How do I count tokens in Python code?",
        "Explain the context window limits for my hardware"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print(f"{'='*70}")

        result = agent.run(query)

        # Display structured result
        print(f"\nQuery Type: {result.get('query_type', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.0%}")
        print(f"Tools Used: {', '.join(result.get('tools_used', [])) or 'None'}")
        print(f"Processing Time: {result.get('processing_time_ms', 0)}ms")
        print(f"\nToken Estimate:")
        te = result.get('token_estimate', {})
        print(f"  Input: {te.get('input_tokens', 0)} tokens")
        print(f"  Output: {te.get('output_tokens', 0)} tokens")
        print(f"  Cost: ${te.get('estimated_cost', 0):.6f}")

        print(f"\nRecommendations:")
        for rec in result.get('recommendations', []):
            print(f"  [{rec.get('priority', 'low').upper()}] {rec.get('title', '')}")
            print(f"    {rec.get('description', '')}")
            print(f"    Estimated time: {rec.get('estimated_hours', 0)} hours")

        print(f"\nAnswer: {result.get('answer', 'N/A')}")

        limitations = result.get('limitations', [])
        if limitations:
            print(f"\nLimitations:")
            for lim in limitations:
                print(f"  - {lim}")

    # Show session stats
    print(f"\n{'='*70}")
    print("SESSION STATISTICS")
    print(f"{'='*70}")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("1. Agent loops integrate multiple tools into coherent responses")
    print("2. Structured outputs ensure reliable parsing of agent responses")
    print("3. Context management prevents conversations from growing unbounded")
    print("4. Prompt caching saves costs for repeated queries")
    print("5. Cost tracking shows the real price of agent operations")
    print("6. Error handling prevents crashes on LLM failures")
    print(f"{'='*70}")


if __name__ == "__main__":
    demonstrate_agent()
