from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GKDConfig, GKDTrainer


def main() -> None:
    data_dir = "artifacts/gkd_data"
    output_dir = "artifacts/gkd_output"

    student_id = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_id = "Qwen/Qwen2.5-7B-Instruct"

    # Load datasets
    train_dataset = load_dataset("json", data_files=f"{data_dir}/train.jsonl")["train"]
    eval_dataset = load_dataset("json", data_files=f"{data_dir}/eval.jsonl")["train"]
    test_dataset = load_dataset("json", data_files=f"{data_dir}/test.jsonl")["train"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Models
    model = AutoModelForCausalLM.from_pretrained(
        student_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # Training configuration
    training_args = GKDConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        eval_steps=200,
        save_steps=400,
        warmup_ratio=0.05,
        fp16=True,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = GKDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Evaluate on test set")
    print(trainer.evaluate(test_dataset))


if __name__ == "__main__":
    main()
