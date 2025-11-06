# === reparse_outputs.py ===
import pandas as pd
import json
import math
import re

# 直接从 <answer> 中抓数值（最快最稳）
_ANS_NUM_RE = re.compile(
    r'<answer>\s*\{\s*"holding_log_delta"\s*:\s*([-+]?\d*\.?\d+)\s*\}\s*</answer>',
    re.DOTALL | re.IGNORECASE,
)

# 若 CSV 把标签转义成 &lt;answer&gt; ... &lt;/answer&gt;
_ANS_NUM_HTML_RE = re.compile(
    r'&lt;answer&gt;\s*\{\s*"holding_log_delta"\s*:\s*([-+]?\d*\.?\d+)\s*\}\s*&lt;/answer&gt;',
    re.DOTALL | re.IGNORECASE,
)

# 退化方案：只在 <answer> 块里再找 JSON（避免误抓 <think>）
_ANS_BLOCK_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r'\{.*?\}', re.DOTALL)

def extract_pred(text: str):
    if not isinstance(text, str) or not text.strip():
        return None

    # 1) 直抓 <answer> 里的数值
    m = _ANS_NUM_RE.search(text)
    if m:
        val = float(m.group(1))
        return val if math.isfinite(val) else None

    # 2) 兼容 HTML 转义的 <answer>
    m = _ANS_NUM_HTML_RE.search(text)
    if m:
        val = float(m.group(1))
        return val if math.isfinite(val) else None

    # 3) 找到 <answer> 块，再在块内尝试 JSON 解析
    block = _ANS_BLOCK_RE.search(text)
    if block:
        ans = block.group(1)
        jm = _JSON_RE.search(ans)
        if jm:
            try:
                obj = json.loads(jm.group(0))
                if "holding_log_delta" in obj:
                    val = float(obj["holding_log_delta"])
                    return val if math.isfinite(val) else None
            except Exception:
                pass
        # 保险：在 <answer> 块里兜底抓一个数字
        mnum = re.search(r'[-+]?\d*\.?\d+', ans)
        if mnum:
            val = float(mnum.group(0))
            return val if math.isfinite(val) else None

    # 4) 完全失败才放弃（绝不去全局抓，避免再误抓 <think> 的 0.03 / 100）
    return None


# ======== 主流程 ========
def reparse_and_update(csv_path: str, save_path: str):
    print(f"🔍 Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    if "raw_output" not in df.columns:
        raise ValueError("❌ CSV 缺少 'raw_output' 列！")

    print("🧠 Re-parsing model outputs...")
    df["parsed_pred"] = df["raw_output"].apply(extract_pred)

    # ✅ 如果原文件包含 parsed_key / parsed_value 列，也可一并更新
    if "parsed_key" in df.columns:
        df["parsed_key"] = "holding_log_delta"
    if "parsed_value" in df.columns:
        df["parsed_value"] = df["parsed_pred"]

    # 覆盖保存或另存为新文件
    df.to_csv(save_path, index=False)
    print(f"✅ Updated CSV saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-parse raw model outputs using updated extract_pred.")
    parser.add_argument("--csv", required=True, help="Path to original CSV file.")
    parser.add_argument("--save", required=True, help="Path to save updated CSV.")
    args = parser.parse_args()

    reparse_and_update(args.csv, args.save)
