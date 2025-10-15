# src/backends/hf_infer.py
# -*- coding: utf-8 -*-
import re, json, math
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_JSON_RE = re.compile(r'\{.*\}', re.S)
_FLOAT_RE = re.compile(r'-?\d+(\.\d+)?([eE][+-]?\d+)?')


def load_samples(fp: str):
    rows = []
    if fp.endswith(".jsonl"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line: rows.append(json.loads(line))
    else:
        rows = json.load(open(fp, "r", encoding="utf-8"))
    return rows


def parse_quarter_from_user(user_text: str) -> str:
    m = re.search(r'Timeline:\s*\(t\)\s*([0-9]{4})-([0-9]{2})-', user_text)
    if not m: return "NA"
    y, mth = int(m.group(1)), int(m.group(2))
    q = (mth-1)//3 + 1
    return f"{y}Q{q}"


def extract_y_true(assistant_content: str):
    try:
        v = float(json.loads(assistant_content)["holding_tp1"])
        return max(v, 0.0)
    except Exception:
        m = _JSON_RE.search(assistant_content)
        if not m: return None
        try:
            v = float(json.loads(m.group(0))["holding_tp1"])
            return max(v, 0.0)
        except Exception:
            return None


def extract_pred(text: str):
    m = _JSON_RE.search(text)
    if m:
        try:
            v = float(json.loads(m.group(0)).get("holding_tp1"))
            if math.isfinite(v): return max(v, 0.0)
        except Exception:
            pass
    m2 = _FLOAT_RE.search(text)
    if not m2: return None
    try:
        v = float(m2.group(0))
        if math.isfinite(v): return max(v, 0.0)
    except Exception:
        return None


def load_model_and_tokenizer(base_model: str, lora_path: str|None, torch_dtype="bfloat16"):
    dtype = torch.bfloat16 if torch_dtype=="bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    if lora_path and lora_path.lower()!="none":
        mdl = PeftModel.from_pretrained(mdl, lora_path)
        try: mdl = mdl.merge_and_unload()
        except Exception: pass
    return tok, mdl


@torch.no_grad()
def infer_chat_batch(tokenizer, model, list_messages: List[List[Dict[str,str]]],
                     max_new_tokens=48, temperature=0.0) -> List[str]:
    prompts=[]
    for msgs in list_messages:
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **inputs,
        do_sample=(temperature>0),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def build_eval_inputs(test_path: str) -> Tuple[List[List[Dict[str,str]]], List[float], List[str], List[int]]:
    rows = load_samples(test_path)
    chat_inputs, y_true, quarter, ids = [], [], [], []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        sys = next((m for m in msgs if m["role"]=="system"), None)
        usr = next((m for m in msgs if m["role"]=="user"), None)
        assistants = [m for m in msgs if m.get("role") == "assistant"]
        if not (sys and usr and assistants):
            continue
        # pick the assistant message that has loss=True if present; otherwise pick the last assistant
        ast = next((m for m in assistants if m.get("loss") is True), assistants[-1])
        yt = extract_y_true(ast.get("content", ""))
        if yt is None: continue
        chat_inputs.append([{"role":"system","content":sys["content"]},
                            {"role":"user","content":usr["content"]}])
        y_true.append(yt)
        quarter.append(parse_quarter_from_user(usr["content"]))
        ids.append(i)
    return chat_inputs, y_true, quarter, ids
