#!/usr/bin/env python3
from llm_finetune_service.config import Settings
from llm_finetune_service.training.train import run_training


if __name__ == "__main__":
    summary = run_training(Settings())
    print(f"Training complete. Adapter written to {summary.adapter_dir}")
