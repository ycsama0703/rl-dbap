from transformers import AutoTokenizer
text = """ <think>As a quantitative portfolio manager analyzing TREDEGAR CORP (TG), here's my assessment of the fundamental shifts from t-1 → t and their directional influence on holdings:\n\n**Key Metric Shifts (t-1 → t):**\n- **me**: 503.4 → 463.7 (-7.9%) - Market equity decline\n- **be**: 300.3 → 109.1 (-63.6%) - Book value collapse  \n- **profit**: 0.328 → 0.906 (+176.8%) - Profitability surge\n- **Gat**: -0.253 → -0.325 (-28.9%) - Growth-at-a-price deterioration\n- **beta**: 1.048 → 0.981 (-6.5%) - Systematic risk reduction\n\n**Directional Influence Analysis:**\nThe dramatic 63.6% collapse in book value combined with a 7.9% market cap decline signals severe fundamental deterioration</think>\n<answer>{"holding_log_delta": -0.01}</answer>"""
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
ids = tok(text)['input_ids']
print(len(ids))
