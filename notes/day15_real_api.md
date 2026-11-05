# Day 15: Real Provider Calls and Honest Usage Metadata

## Goal

Connect the learning assistant to a real model service while keeping provider failures, token counts, and cost estimates explicit. This is a learning experiment, not production-ready infrastructure.

## Concept: provider versus agent

A **provider** sends messages to one model service and normalizes its response. The **agent** prepares messages, calls a provider chain, validates the response, and reports metadata. Keeping those responsibilities separate makes it possible to compare local and cloud inference without hiding provider differences.

A fallback chain is resilience logic, not a quality guarantee. The fallback may use a different model and may have different latency, behavior, and billing. This demo defaults to one local provider; cloud calls or automatic fallback must be selected explicitly.

## What the experiment does

- `experiments/15_real_agent_api.py` has Anthropic, OpenAI, and Ollama provider adapters.
- Ollama is called over its local HTTP API. If the server or configured model is unavailable, the request fails; there is no simulated answer.
- `AI_LAB_PROVIDER` selects `ollama` (default), `anthropic`, `openai`, or `auto`. `auto` tries Anthropic, then OpenAI, then Ollama, so it may make a paid call.
- One run prompts for one question. Transient retries or `auto` fallback can send multiple provider requests; use a single explicitly selected provider to avoid cross-provider fallback.
- The response is parsed as JSON and checked for required fields and basic types. This is application-side validation; asking for a schema in the prompt does not force every model to obey it.
- Token usage is kept as `provider_reported`, `partially_reported`, or `unavailable`. Missing counts are `null`, never fabricated as zero or a character-based estimate.
- Local inference cost is reported as `$0` in provider API fees, not as zero electricity or hardware cost. Cloud cost is unavailable unless the user supplies rates.
- Optional rates are USD per million tokens:
  - `OPENAI_INPUT_USD_PER_MILLION`, `OPENAI_OUTPUT_USD_PER_MILLION`
  - `ANTHROPIC_INPUT_USD_PER_MILLION`, `ANTHROPIC_OUTPUT_USD_PER_MILLION`

## Configuration (PowerShell)

```powershell
# Local run (default); install/start Ollama and pull the configured model first.
$env:AI_LAB_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "qwen3.5:4b"
# Qwen thinking is disabled by default for faster structured answers.
# Set to true when you want to experiment with thinking output (may be slower).
$env:OLLAMA_THINK = "false"
python experiments/15_real_agent_api.py

# Explicit cloud provider. Do not paste keys into source files.
$env:AI_LAB_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY = "your-key"
$env:ANTHROPIC_MODEL = "your-enabled-model-id"
python experiments/15_real_agent_api.py
```

For OpenAI, set `AI_LAB_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. For Anthropic, both `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` are required. Cloud model IDs are deliberately not hard-coded because provider catalogs change. `openai` is optional and is not currently listed in `requirements.txt`; install it only if you choose that provider. Use model IDs and rates available to your account rather than relying on values in old notes.

`OLLAMA_HOST` can change the local server URL. `OLLAMA_THINK` defaults to `false`; set it to `true` to request model thinking when supported. Thinking can greatly increase latency on CPU/GPU-offloaded hardware. The local model must already be present in that server. Keep API keys in environment variables and do not commit them.

## Reading usage and cost

An API-reported token count is the provider's count for that request. A missing count is not zero. This script does not claim to know coding-assistant subscription credits, hidden reasoning usage, or billing data the provider does not return.

Cloud cost is an estimate from user-configured input/output rates and reported token counts. It may omit cached-token rates, tools, taxes, or other billing rules. If rate configuration or full usage is missing, cost is returned as unavailable.

## Limitations and next steps

- There is no actual tool execution yet: tools listed in earlier drafts were only descriptions. Add one narrow, deterministic tool and an explicit tool-call loop after this provider layer is understood.
- JSON field validation is a small hand-written subset, not a full JSON Schema validator. It checks the fields this demo uses.
- The provider model defaults are examples. Override them with a model ID currently enabled for your account or installed in Ollama.
- This demo has no conversation history, streaming, retries for malformed model output, or provider health scoring.
- Next: run one local query, inspect the returned usage source and latency, then compare with one explicitly selected cloud provider only if desired. Record model ID, date, prompt, output quality, reported tokens, and measured latency.
