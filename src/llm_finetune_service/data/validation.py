from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

FORMAL_PHRASES = [
    "i am writing to inform you",
    "please be advised",
    "kind regards",
    "thank you for your attention",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate record ids detected.")

    required_fields = {"id", "split", "scenario_type", "instruction", "source_email", "target_slack", "metadata"}
    for record in records:
        missing = required_fields.difference(record)
        if missing:
            errors.append(f"Record {record.get('id', '<missing-id>')} missing fields: {sorted(missing)}")
        if record.get("split") not in {"train", "validation", "test"}:
            errors.append(f"Record {record.get('id')} has invalid split {record.get('split')!r}")
        if len(record.get("target_slack", "")) >= len(record.get("source_email", "")):
            errors.append(f"Record {record.get('id')} target is not shorter than source.")
        lowered_target = record.get("target_slack", "").lower()
        if any(phrase in lowered_target for phrase in FORMAL_PHRASES):
            errors.append(f"Record {record.get('id')} target still contains formal phrasing.")

    duplicate_sources = [text for text, count in Counter(normalize(record["source_email"]) for record in records).items() if count > 1]
    if duplicate_sources:
        errors.append("Exact duplicate source_email values detected.")

    duplicate_pairs = [
        pair
        for pair, count in Counter(
            (normalize(record["source_email"]), normalize(record["target_slack"])) for record in records
        ).items()
        if count > 1
    ]
    if duplicate_pairs:
        errors.append("Exact duplicate source/target pairs detected.")

    return errors


def validate_dataset_dir(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    all_records: list[dict[str, Any]] = []
    seen_sources_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}

    for split_name in ("train", "validation", "test"):
        path = dataset_dir / f"{split_name}.jsonl"
        if not path.exists():
            errors.append(f"Missing split file: {path}")
            continue
        records = load_jsonl(path)
        if not records:
            errors.append(f"Split {split_name} is empty.")
        split_errors = validate_records(records)
        errors.extend(split_errors)
        all_records.extend(records)
        seen_sources_by_split[split_name] = {normalize(record["source_email"]) for record in records}

    if all_records:
        train_sources = seen_sources_by_split["train"]
        validation_sources = seen_sources_by_split["validation"]
        test_sources = seen_sources_by_split["test"]
        if train_sources.intersection(validation_sources):
            errors.append("Train/validation leakage detected in source_email.")
        if train_sources.intersection(test_sources):
            errors.append("Train/test leakage detected in source_email.")
        if validation_sources.intersection(test_sources):
            errors.append("Validation/test leakage detected in source_email.")

        style_ratio = sum(
            any(
                marker in record["target_slack"].lower()
                for marker in [
                    "quick",
                    "heads up",
                    "fyi",
                    "clarify",
                    "can we",
                    "good news",
                    "approved",
                    "incident update",
                    "vendor update",
                    "feedback",
                    "reminder",
                ]
            )
            for record in all_records
        ) / len(all_records)
        if style_ratio < 0.45:
            errors.append("Slack-style markers are too sparse across the dataset.")

    return errors
