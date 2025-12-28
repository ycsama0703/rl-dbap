#!/usr/bin/env python
"""
Minimal DeepSeek API call tester.
Uses DEEPSEEK_API_KEY (or OPENAI_API_KEY) and DEEPSEEK_API_BASE/DEEPSEEK_MODEL if set.
"""

from __future__ import annotations

import os
from openai import OpenAI


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY/OPENAI_API_KEY.")
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=api_base)
    prompt = "Say hello briefly."
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.2,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
