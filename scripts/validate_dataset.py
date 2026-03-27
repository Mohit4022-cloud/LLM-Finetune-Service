#!/usr/bin/env python3
from llm_finetune_service.config import Settings
from llm_finetune_service.data.validation import validate_dataset_dir


if __name__ == "__main__":
    settings = Settings()
    errors = validate_dataset_dir(settings.dataset_dir)
    if errors:
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Dataset at {settings.dataset_dir} passed validation.")
