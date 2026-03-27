#!/usr/bin/env python3
from llm_finetune_service.config import Settings
from llm_finetune_service.data.dataset_builder import build_dataset


if __name__ == "__main__":
    result = build_dataset(Settings())
    print(f"Built dataset with split counts: {result.split_counts}")
