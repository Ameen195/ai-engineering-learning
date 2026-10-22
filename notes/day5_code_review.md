# Day 5: Code Review Learning Project

## Deployed browser demo: Review Bench

The Vercel site is a static page. Its browser-side checks run locally on the source text; the page does not send source to a backend or AI provider, and it does not need a Groq key or demo password.

The visual interface is a developer workbench: source editor at left, findings at right, with severity, explanation, recommendation, and a jump-to-line control.

### Rules checked

- Calls to `eval()` or `exec()` — high severity pattern warning.
- Bare `except:` handlers — medium severity pattern warning.
- One-line `subprocess.run()`, `Popen()`, or `call()` calls with `shell=True` — high severity pattern warning.

These are intentionally simple line-level heuristics, not Python AST analysis, an LLM review, a complete scanner, or a security audit. They can miss multi-line patterns, false-positive on harmless code, and cannot reason about program context. No match does not mean the code is safe.

### Run and test the static demo

Open `public/index.html` in a browser or deploy the `public` directory as a static Vercel site. To run checks:

```powershell
npm run test:review
```

The browser demo can load a local `.py` file or accept pasted source up to 25,000 bytes. It does not retain source after the page is closed.

### Vercel deployment

Use the `Other` preset, leave build and install commands empty, and select `public` as the output directory. No Vercel environment variables are needed for the static demo. Hosting still follows the account's plan limits and terms; static hosting avoids Groq/model inference charges but does not guarantee every account's total cost is zero.

## Preserved API experiments (not deployed)

- `experiments/05_code_review_agent.py` retains a local Python AST checker and optional Claude CLI code path.
- `experiments/groq_vercel_api_reference.py` preserves the earlier Groq HTTP handler for learning; Vercel no longer deploys it as a function.
- Python tests for that reference use mocked provider responses. They do not make network requests or consume API credits.

## Safety and privacy boundaries

- Reviewed source is not executed by the static browser checker.
- The static browser checker makes no network call for review.
- The separate Claude CLI and Groq reference integrations send source to those providers only when explicitly configured and invoked. Provider policies and charges apply; don't submit confidential code without authorization.
- No model/API usage is claimed or measured by the static page.
