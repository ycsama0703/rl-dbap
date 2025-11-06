"""DeepSeek chat helper without external SDK dependencies."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests


API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")


GENERATOR_PROMPT = """
You are an experienced quantitative portfolio manager.

Given the historical fundamentals and the latest observation of a company,
reason numerically about how the holdings should change. Present the thinking
inside a <think> block, then provide the predicted holding_log_delta inside the
<answer> block.

Strict output contract:
1) The <think> block must contain 3-5 concise sentences referencing t-1→t
   changes for me, be, profit, Gat, beta, plus any portfolio constraints.
2) Immediately after </think>, emit:
   <answer>{{"holding_log_delta": <float>}}</answer>
   - The float must be finite, with up to 6 decimals, no scientific notation.
3) Output nothing beyond these two blocks; stop right after </answer>.

Input prompt:
---
{user_prompt}
"""


def _post_chat(messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> Dict[str, Any]:
    url = API_BASE.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
    resp.raise_for_status()
    return resp.json()


def generate_assistant(user_text: str, temperature: float = 0.7, max_tokens: int = 320) -> str:
    prompt = GENERATOR_PROMPT.format(user_prompt=user_text)
    messages = [
        {"role": "system", "content": "You are a financial reasoning assistant."},
        {"role": "user", "content": prompt},
    ]
    data = _post_chat(messages, temperature=temperature, max_tokens=max_tokens)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected response: {json.dumps(data, ensure_ascii=False)}")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return content.strip()


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Call DeepSeek chat via scripts/deepseek.py")
    parser.add_argument("--prompt", help="User prompt text; read stdin if omitted.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=320)
    args = parser.parse_args()

    if args.prompt:
        prompt_text = args.prompt
    else:
        if sys.stdin.isatty():
            parser.error("Provide --prompt or pipe text via stdin.")
        prompt_text = sys.stdin.read()

    output = generate_assistant(prompt_text, temperature=args.temperature, max_tokens=args.max_tokens)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

