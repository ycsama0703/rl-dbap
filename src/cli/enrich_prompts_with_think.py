import os
import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Prompt 模板
PROMPT_TEMPLATE = """
You are an experienced quantitative portfolio manager.

Given the historical data below, write a concise reasoning (no more than 3 sentences) inside <think>...</think>.
- Focus on key metric changes (me, be, profit, Gat, beta) from t-1 → t.
- Keep it short and numeric: express only main percentage changes and one-line interpretation.
- Avoid repeating values already shown in the data.
- End with the directional conclusion for holdings (increase/decrease/mild change).
Return only the <think> block, nothing else.

---
{user_prompt}
"""


def generate_think_deepseek(prompt_text: str) -> str:
    """调用 DeepSeek 生成推理文本"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a financial reasoning assistant."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(user_prompt=prompt_text)}
            ],
            temperature=0.7,
            max_tokens=250,
        )
        out = resp.choices[0].message.content.strip()
        if not out.startswith("<think>"):
            out = f"<think>{out}</think>"
        return out
    except Exception as e:
        print(f"[WARN] DeepSeek generation failed: {e}")
        return "<think>[FAILED TO GENERATE]</think>"


def enrich_file(in_path: Path, out_path: Path, limit: int = None):
    """对单个 jsonl 文件进行 enrich，增加 <think> 字段"""
    lines = in_path.read_text(encoding="utf-8").splitlines()
    total = len(lines) if limit is None else min(limit, len(lines))
    success, fail = 0, 0

    with out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(lines[:total], desc=f"  └─ {in_path.name}", leave=False, unit="sample"):
            try:
                rec = json.loads(line)
                rec["think"] = generate_think_deepseek(rec["prompt"])
                success += 1
            except Exception as e:
                rec = {"error": str(e)}
                fail += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ {in_path.name}: success={success}, fail={fail}, total={total}")


if __name__ == "__main__":
    src_dir = Path("artifacts/prompts_hist_sft")
    dst_dir = Path("artifacts/prompts_hist_sft_with_think")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 可选参数：只跑前 N 条（默认 10 条验证）
    LIMIT = 10

    print(f"\n🚀 Enriching prompts from {src_dir} -> {dst_dir}")
    print(f"🧠 Using DeepSeek API: {client.base_url}")
    print(f"⚙️  Generating up to {LIMIT} samples per file\n")

    for f in tqdm(list(src_dir.glob("*.jsonl")), desc="Files", unit="file"):
        enrich_file(f, dst_dir / f.name, limit=LIMIT)

    print("\n🎯 All done! DeepSeek-enriched prompts saved to:")
    print(dst_dir.resolve())
