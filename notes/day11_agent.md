# Day 11: Building Your First Complete Agent

## Agent Architecture

```
User Query → [Agent Loop] → [Tool Selection] → [Execute Tools] → [LLM Response]
                ↑                                   ↓
                └────── Structured Output ←────────┘
```

**Agent Loop:**
1. Parse user query
2. Decide what tools to use
3. Execute tools and collect results
4. Generate structured response using LLM
5. Repeat if needed (multi-step tasks)

## Key Components

### 1. Conversation Management
```python
self.conversation_history = []
self.max_context_tokens = 4000

def _manage_context(self):
    """Keep conversation within token budget."""
    total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
    if total_chars > 12000:  # ~3000 tokens threshold
        self.conversation_history = self.conversation_history[2:]
```

**Why:** Prevents context snowball (from Day 2). Each turn, check if conversation exceeds token budget. Trim oldest messages if needed.

### 2. Structured Responses
```python
AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string"},
        "confidence": {"type": "number"},
        "tools_used": {"type": "array", "items": {"type": "string"}},
        ...
    }
}
```

**Why:** Eliminates parsing errors (from Day 10). Agent always returns predictable JSON structure.

### 3. Prompt Caching
```python
def _generate_cache_key(self, query: str) -> str:
    return str(hash(query[:100]))

if cache_key in self.cache:
    return cached_response  # Free!
```

**Why:** Repeated questions get cached (from Day 3). Saves API costs.

### 4. Tool Integration
Your agent uses these integrated tools:

| Tool | From Day | Purpose |
|------|----------|---------|
| `calculate_vram()` | Day 8 | Check model hardware compatibility |
| `estimate_tokens()` | Day 1 | Estimate prompt length |
| `find_compatible_models()` | Day 4 | Compare local/cloud options |
| `json schema validation` | Day 10 | Ensure structured responses |

### 5. Error Handling
```python
try:
    result = json.loads(content)
except json.JSONDecodeError as e:
    return {"error": "Invalid JSON from LLM"}
```

**Why:** LLMs sometimes return malformed responses. Graceful error handling prevents crashes.

## Cost Tracking

```python
# Track session costs
self.total_cost = 0.0
self.total_tokens = 0

# Estimate per call
cost = (input_tokens / 1000) * 0.003 + (output_tokens / 1000) * 0.006
```

**Why:** Agents can be expensive (from Day 1-2 lessons on token costs). Track usage to optimize.

## Design Decisions

1. **Temperature 0.3**: Slightly deterministic for technical responses (not creative)
2. **Hardware-aware**: Agent knows your GTX 1650 constraints
3. **Schema validation**: Ensures reliable parsing
4. **Cache-first**: Saves costs on repeated questions
5. **Graceful degradation**: Falls back to error responses on failure

## Your Learning Journey So Far

| Day | Topic | Key Experiment |
|-----|-------|----------------|
| 1 | Tokenization | Tokenizer comparison |
| 2 | Context windows | Token counting & context snowball |
| 3 | Prompt caching | Cost optimization |
| 4 | Model comparison | Performance metrics |
| 5 | Code review | Static demo project |
| 6-7 | Transformers | Attention & architecture |
| 8 | VRAM | Hardware optimization |
| 9 | Agent patterns | Agent loop simulation |
| 10 | API patterns | Real API integration |
| 11 | Complete agent | Learning Assistant Agent |

Each day's learning integrates into the final agent, demonstrating how foundational concepts combine into real AI systems.

## Files Created

- `experiments/11_learning_assistant_agent.py` - Complete agent implementation
- `notes/day11_agent.md` - This file

Run: `python experiments/11_learning_assistant_agent.py`
