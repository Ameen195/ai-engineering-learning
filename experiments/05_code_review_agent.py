"""Review one Python file with Claude and print structured findings."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from review_core import MAX_SOURCE_BYTES, build_prompt, local_static_review, parse_findings, render_findings

DEFAULT_MODEL = "claude-sonnet-5-5"


def review_source(source: str, client: Any, model: str) -> tuple[list[dict[str, Any]], Any, float]:
    """Call Claude once, validate its response, and return findings, usage, latency."""
    started_at = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        messages=[{"role": "user", "content": build_prompt(source)}],
    )
    latency_seconds = time.perf_counter() - started_at
    text = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )
    findings = parse_findings(text, len(source.splitlines()))
    return findings, getattr(response, "usage", None), latency_seconds


def load_source(path: Path) -> str:
    """Read a single UTF-8 Python file without executing it."""
    if path.suffix.lower() != ".py":
        raise ValueError("Input must be a .py file")
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"File exceeds the {MAX_SOURCE_BYTES:,}-byte input limit")
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Review one Python file with local checks or Claude.")
    parser.add_argument("file", type=Path, help="Path to one UTF-8 .py file")
    parser.add_argument("--claude", action="store_true", help="Also request an AI review (uses API credits)")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    try:
        source = load_source(args.file)
        findings = local_static_review(source)
        print("LOCAL STATIC CHECKS (limited rules; not a complete security audit)")
        print(render_findings(findings))

        if args.claude:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("Set ANTHROPIC_API_KEY before using --claude.", file=sys.stderr)
                return 2
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
            findings, usage, latency_seconds = review_source(source, client, args.model)
            print("\nCLAUDE REVIEW (model suggestions; verify before acting)")
            print(render_findings(findings))
            print(f"\nModel: {args.model}")
            print(f"Measured request latency: {latency_seconds:.2f} seconds")
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if input_tokens is not None:
                    print(f"Provider-reported input tokens: {input_tokens}")
                if output_tokens is not None:
                    print(f"Provider-reported output tokens: {output_tokens}")
        else:
            print("\nNo model API call was made. Add --claude to request an AI review.")
        return 0
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not review file: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Review request failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
