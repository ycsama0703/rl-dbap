import argparse
from pathlib import Path
from typing import Optional
import json
import shutil
import tempfile

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GKDConfig, GKDTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GKD distillation with optional LoRA teacher")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing train.jsonl / eval.jsonl / test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store distilled model and tokenizer",
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="Path or identifier of the student base model",
    )
    parser.add_argument(
        "--teacher-base",
        type=str,
        required=True,
        help="Path or identifier of the teacher base model",
    )
    parser.add_argument(
        "--teacher-lora",
        type=str,
        default=None,
        help="Optional path to teacher LoRA checkpoint (adapter config + weights)",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=400)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true", default=False)
    return parser.parse_args()


def load_teacher(teacher_base: str, lora_path: Optional[str]):
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_base,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path:
        lora_path = Path(lora_path)
        tmp_dir = Path(tempfile.mkdtemp(prefix="gkd_lora_"))
        for src in lora_path.iterdir():
            dest = tmp_dir / src.name
            if src.name == "adapter_config.json":
                with src.open("r", encoding="utf-8") as f:
                    config = json.load(f)
                config.pop("corda_config", None)
                with dest.open("w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
            else:
                shutil.copy2(src, dest)
        teacher_model = PeftModel.from_pretrained(teacher_model, str(tmp_dir))
        teacher_model = teacher_model.merge_and_unload()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return teacher_model


def main() -> None:
    args = parse_args()

    train_dataset = load_dataset("json", data_files=f"{args.data_dir}/train.jsonl")["train"]
    eval_dataset = load_dataset("json", data_files=f"{args.data_dir}/eval.jsonl")["train"]
    test_dataset = load_dataset("json", data_files=f"{args.data_dir}/test.jsonl")["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    teacher_model = load_teacher(args.teacher_base, args.teacher_lora)

    training_args = GKDConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("Evaluate on test set")
    print(trainer.evaluate(test_dataset))


if __name__ == "__main__":
    main()
