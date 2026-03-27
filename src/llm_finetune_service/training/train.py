from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_finetune_service.config import Settings
from llm_finetune_service.training.prompts import render_prompt


@dataclass(slots=True)
class TrainingRunSummary:
    run_dir: Path
    adapter_dir: Path
    metrics_path: Path
    train_config_path: Path


def _detect_device() -> tuple[str, bool]:
    import torch

    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", True
    return "cpu", False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_training(settings: Settings | None = None) -> TrainingRunSummary:
    settings = settings or Settings()
    if not settings.train_split_path.exists():
        raise FileNotFoundError(
            f"Training split not found at {settings.train_split_path}. Run the dataset build step first."
        )
    if not settings.validation_split_path.exists():
        raise FileNotFoundError(
            f"Validation split not found at {settings.validation_split_path}. Run the dataset build step first."
        )

    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    device, has_accelerator = _detect_device()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("artifacts/train_runs") / run_id
    adapter_dir = run_dir / "adapter"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(settings.train_split_path),
            "validation": str(settings.validation_split_path),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(record: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt(record, include_target=False)
        full_text = render_prompt(record, include_target=True)
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=settings.max_input_tokens)
        full_tokens = tokenizer(full_text, truncation=True, max_length=settings.max_input_tokens)
        labels = list(full_tokens["input_ids"])
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        full_tokens["labels"] = labels
        return full_tokens

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    model_kwargs: dict[str, Any] = {}
    if device == "cuda":
        import torch

        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(settings.model_name, **model_kwargs)
    lora_config = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=settings.learning_rate,
        logging_steps=10,
        num_train_epochs=settings.training_epochs,
        max_steps=settings.max_steps,
        eval_strategy="steps",
        eval_steps=max(10, settings.max_steps // 4),
        save_strategy="steps",
        save_steps=settings.max_steps,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        fp16=device == "cuda",
        use_mps_device=device == "mps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    train_result = trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    _write_json(
        run_dir / "train_config.json",
        {
            "model_name": settings.model_name,
            "train_split_path": str(settings.train_split_path),
            "validation_split_path": str(settings.validation_split_path),
            "training_epochs": settings.training_epochs,
            "max_steps": settings.max_steps,
            "learning_rate": settings.learning_rate,
            "lora_r": settings.lora_r,
            "lora_alpha": settings.lora_alpha,
            "lora_dropout": settings.lora_dropout,
            "device": device,
            "accelerator_available": has_accelerator,
        },
    )
    _write_json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "timestamp": run_id,
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "train_runtime_seconds": train_result.metrics.get("train_runtime"),
            "train_loss": train_result.metrics.get("train_loss"),
            "eval_loss": metrics.get("eval_loss"),
            "global_step": train_result.metrics.get("global_step"),
        },
    )
    _write_json(
        run_dir / "sample_predictions.json",
        {
            "note": "Populate with qualitative generations after running evaluation.",
            "generated": [],
        },
    )

    latest_pointer = Path("artifacts/train_runs/latest")
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    if latest_pointer.exists() or latest_pointer.is_symlink():
        latest_pointer.unlink()
    latest_pointer.symlink_to(run_dir.name)

    return TrainingRunSummary(
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        metrics_path=run_dir / "metrics.json",
        train_config_path=run_dir / "train_config.json",
    )
