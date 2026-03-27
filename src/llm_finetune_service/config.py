from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


InferenceMode = Literal["base", "adapter", "dev_fallback"]


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_path: Path = Path(os.getenv("ADAPTER_PATH", "artifacts/train_runs/latest/adapter"))
    inference_mode: InferenceMode = os.getenv("INFERENCE_MODE", "base")  # type: ignore[assignment]
    allow_dev_fallback: bool = _get_bool("ALLOW_DEV_FALLBACK", False)
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", "data/splits"))
    train_split_path: Path = Path(os.getenv("TRAIN_SPLIT_PATH", "data/splits/train.jsonl"))
    validation_split_path: Path = Path(os.getenv("VALIDATION_SPLIT_PATH", "data/splits/validation.jsonl"))
    test_split_path: Path = Path(os.getenv("TEST_SPLIT_PATH", "data/splits/test.jsonl"))
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "512"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "120"))
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_ttl: int = int(os.getenv("REDIS_TTL", "3600"))
    data_seed: int = int(os.getenv("DATASET_SEED", "42"))
    dataset_size: int = int(os.getenv("DATASET_SIZE", "480"))
    training_epochs: int = int(os.getenv("TRAINING_EPOCHS", "1"))
    max_steps: int = int(os.getenv("MAX_STEPS", "120"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    eval_limit: int = int(os.getenv("EVAL_LIMIT", "0"))


def get_settings() -> Settings:
    return Settings()
