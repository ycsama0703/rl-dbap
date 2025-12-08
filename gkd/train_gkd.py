#!/usr/bin/env python
"""
Lightweight wrapper to run TRL GKD distillation on our chat datasets.

The script expects JSONL files with a `messages` field (system/user/assistant) like our SFT/GRPO data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GKDConfig, GKDTrainer


def _load_split(path: str, split: str = "train"):
    ds = load_dataset("json", data_files=path)[split]
    keep_cols = {"messages"}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    return ds.remove_columns(drop_cols)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run GKD distillation on chat JSONL data.")
    ap.add_argument("--student", required=True, help="Path or HF id of student model")
    ap.add_argument("--teacher", required=True, help="Path or HF id of teacher model")
    ap.add_argument("--train-path", required=True, help="JSONL with `messages` for training")
    ap.add_argument("--eval-path", help="Optional JSONL with `messages` for eval/val")
    ap.add_argument("--output-dir", default="gkd-model")
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--per-device-eval-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16")
    return ap.parse_args()


def load_model(path: str, tok=None, device_map: str | None = None):
    """Load a model; if tok is provided, reuse it, otherwise load a tokenizer."""
    local_tok = tok or AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
    if local_tok.pad_token_id is None:
        local_tok.pad_token = local_tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device_map,
    )
    return local_tok, mdl


def main():
    args = parse_args()
    train_ds = _load_split(args.train_path, "train")
    eval_ds = _load_split(args.eval_path, "train") if args.eval_path else None

    # Use a shared tokenizer to align vocab sizes between student and teacher.
    shared_tok, teacher_mdl = load_model(args.teacher, tok=None, device_map="auto")
    student_tok, student_mdl = load_model(args.student, tok=shared_tok, device_map="auto")

    # Ensure vocab sizes match tokenizer length to avoid logits shape mismatch.
    vocab_len = len(shared_tok)
    if student_mdl.get_input_embeddings().weight.shape[0] != vocab_len:
        student_mdl.resize_token_embeddings(vocab_len)
    if teacher_mdl.get_input_embeddings().weight.shape[0] != vocab_len:
        teacher_mdl.resize_token_embeddings(vocab_len)

    gkd_cfg = GKDConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
    )

    trainer = GKDTrainer(
        model=student_mdl,
        teacher_model=teacher_mdl,
        args=gkd_cfg,
        processing_class=student_tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    student_tok.save_pretrained(Path(args.output_dir))


if __name__ == "__main__":
    main()
