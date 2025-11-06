from pathlib import Path

path = Path("scripts/preview_banks_dataset.py")
text = path.read_text(encoding="utf-8")
start = text.index("def _rewrite_user")
end = text.index("def _call_deepseek")
new_block = """def _rewrite_user(original: str) -> str:\n    head, *_ = original.partition(\"\\n\\nOUTPUT FORMAT\")\n    head = head.strip()\n    instructions = (\n        \"\\n\\nRESPONSE FORMAT\\n\"\n        \"- Provide one <think>...</think> block with 3-5 concise sentences highlighting t-1->t numeric deltas.\\n\"\n        \"- Immediately follow with <answer>{\\\"holding_log_delta\\\": <float>}</answer> using 2 decimal places and no extra text.\\n\"\n        \"- Stop right after </answer>.\"\n    )\n    return head + instructions\n\n\n"""
path.write_text(text[:start] + new_block + text[end:], encoding="utf-8")
