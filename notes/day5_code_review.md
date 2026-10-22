# Day 5: First Code Review Agent

## Goal and current MVP

Build a read-only reviewer for one Python file. The project has three ways to use it:

1. Local CLI static checks using Python's AST: reports syntax errors, `eval()`/`exec()`, bare `except`, and subprocess calls with `shell=True`.
2. Optional Claude CLI review: with `--claude`, sends one file to Anthropic and validates structured findings.
3. Private Vercel web app: static upload/paste page calls a password-gated Python function, which sends one file to Groq and validates structured findings.

The app does not execute code or persist submitted source. Local rules are intentionally limited and are not a full security audit. The password is an app-level gate, not identity authentication; enable Vercel deployment protection as an additional layer where available.

## Concept: model output needs a boundary

A model can suggest issues, but its response is ordinary generated text until the program validates it. We ask for JSON, parse it, check each finding's fields/severity/line, and only then render it. This makes the *shape* predictable; it does not prove a finding is correct or complete.

The source file is marked as untrusted input in the prompt. This is a useful instruction, not a security guarantee. We also limit the input size and never execute reviewed code.

## Run it

Set the API key in the current terminal only; do not put it in a source file or commit it.

Run local static checks without a key or API request:

```powershell
python experiments/05_code_review_agent.py experiments/samples/review_me.py
```

For an optional live review, set the API key in the current terminal only, then opt in with `--claude`:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
python experiments/05_code_review_agent.py experiments/samples/review_me.py --claude
```

This sends the source to Anthropic and uses API credits. Do not submit confidential code unless permitted.

Optional model override:
```powershell
$env:ANTHROPIC_MODEL = "model-id-from-current-Anthropic-docs"
```

The default model ID is `claude-sonnet-5-5` as listed in Anthropic's model overview when this experiment was created. Model names and availability can change; check current provider docs before running.

Local checks do not need the Anthropic SDK at runtime or an API key. The SDK is only needed for `--claude` mode.

## Run offline tests

```powershell
python -m unittest discover -s tests -v
```

Tests use a mocked API response, need no API key/network, and do not incur model charges.

## What is measured and what is not

- Request latency is measured locally with a monotonic timer.
- Input/output usage is printed only when returned by the API.
- The script does not report a billing amount or estimate one.
- Findings are suggestions. Verify them against the source and tests.

## Limitations

- A single model call has no agent/tool loop yet.
- It reviews only one UTF-8 `.py` file up to 25,000 bytes.
- It may miss defects, misunderstand context, or return false positives.
- Line references and JSON structure are checked, but semantic correctness is not.
- The Claude CLI sends source to Anthropic; the Vercel web app sends source to Groq. Do not submit confidential code unless permitted by your organization and the provider's current data policies.
- The web app does not write source to an application database or file, but this does not determine provider-side retention or processing.
- No caching is requested or assumed in this first stage.

## Deploy the private Groq web demo

1. Push the app files to the GitHub repository and import that repository into Vercel.
2. Set these server-side environment variables in Vercel Project Settings → Environment Variables:
   - `GROQ_API_KEY`: your Groq key
   - `APP_PASSWORD`: a unique password of at least 20 characters
   - `GROQ_MODEL`: optional; defaults to `openai/gpt-oss-20b`
3. Enable Vercel Deployment Protection before using the deployed endpoint. The shared password gate alone is not account authentication and does not include rate limiting.
4. Redeploy after adding variables. Open the protected deployment URL, enter the app password, and try `experiments/samples/review_me.py`.
5. Check Groq's Console for current model availability, rate limits, and billing. An API key does not guarantee free usage.

The Groq key is sent only from the Vercel server to Groq; it is not included in the browser. The browser sends Python source to the Vercel endpoint, which forwards it to Groq for inference.

## Key terms

- **Structured output**: model text requested in a known data format, here JSON.
- **Validation**: deterministic checks that the response matches the expected fields and types.
- **False positive**: a reported issue that is not actually a defect.
- **False negative**: a real defect the model does not report.
