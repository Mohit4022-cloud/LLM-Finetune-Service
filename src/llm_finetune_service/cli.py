from __future__ import annotations

import argparse
from pathlib import Path

from llm_finetune_service.config import Settings
from llm_finetune_service.data.dataset_builder import build_dataset
from llm_finetune_service.data.validation import validate_dataset_dir
from llm_finetune_service.eval.run_eval import run_evaluation
from llm_finetune_service.training.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-finetune-service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-dataset")
    subparsers.add_parser("validate-dataset")
    subparsers.add_parser("train")
    subparsers.add_parser("evaluate")

    args = parser.parse_args()
    settings = Settings()

    if args.command == "build-dataset":
        result = build_dataset(settings)
        print(f"Built dataset with split counts: {result.split_counts}")
    elif args.command == "validate-dataset":
        errors = validate_dataset_dir(settings.dataset_dir)
        if errors:
            for error in errors:
                print(f"- {error}")
            raise SystemExit(1)
        print(f"Dataset at {settings.dataset_dir} passed validation.")
    elif args.command == "train":
        summary = run_training(settings)
        print(f"Training completed. Adapter saved to {summary.adapter_dir}")
    elif args.command == "evaluate":
        run_dir = run_evaluation(settings)
        print(f"Evaluation completed. Results saved to {run_dir}")
