#!/usr/bin/env python3
from llm_finetune_service.config import Settings
from llm_finetune_service.data.validation import load_jsonl


if __name__ == "__main__":
    settings = Settings()
    for split in ("train", "validation", "test"):
        path = settings.dataset_dir / f"{split}.jsonl"
        records = load_jsonl(path)
        sample = records[0]
        print(f"\n[{split}] {sample['id']} / {sample['scenario_type']}")
        print(f"source: {sample['source_email']}")
        print(f"target: {sample['target_slack']}")
