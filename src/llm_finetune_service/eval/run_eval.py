from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llm_finetune_service.config import Settings
from llm_finetune_service.data.validation import load_jsonl
from llm_finetune_service.eval.metrics import score_prediction, summarize_scores
from llm_finetune_service.inference.model import TextGenerator


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "This report was generated from the held-out test split. It compares the base model against a fine-tuned adapter.",
        "",
        "## Aggregate metrics",
        "",
        f"- Base overall score: {results['aggregate']['base']['overall']}",
        f"- Adapter overall score: {results['aggregate']['adapter']['overall']}",
        "",
        "## Qualitative examples",
        "",
    ]
    for example in results["examples"][:5]:
        lines.extend(
            [
                f"### {example['id']}",
                "",
                f"**Source email**: {example['source_email']}",
                "",
                f"**Reference**: {example['reference']}",
                "",
                f"**Base output**: {example['base_prediction']}",
                "",
                f"**Adapter output**: {example['adapter_prediction']}",
                "",
                f"**Base score**: {example['base_scores']['overall']}",
                "",
                f"**Adapter score**: {example['adapter_scores']['overall']}",
                "",
            ]
        )
    return "\n".join(lines)


def run_evaluation(settings: Settings | None = None) -> Path:
    settings = settings or Settings()
    if not settings.test_split_path.exists():
        raise FileNotFoundError(f"Test split not found at {settings.test_split_path}. Build the dataset first.")
    if not settings.adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter path not found at {settings.adapter_path}. Train a real adapter before running evaluation."
        )

    test_records = load_jsonl(settings.test_split_path)
    if settings.eval_limit > 0:
        test_records = test_records[: settings.eval_limit]

    base_settings = Settings()
    base_settings.inference_mode = "base"
    base_generator = TextGenerator(base_settings)

    adapter_settings = Settings()
    adapter_settings.inference_mode = "adapter"
    adapter_generator = TextGenerator(adapter_settings)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("artifacts/eval") / run_id

    examples: list[dict[str, Any]] = []
    base_scores: list[dict[str, Any]] = []
    adapter_scores: list[dict[str, Any]] = []

    for record in test_records:
        base_prediction = base_generator.generate(record["source_email"])
        adapter_prediction = adapter_generator.generate(record["source_email"])
        base_score = score_prediction(record["source_email"], base_prediction, record["target_slack"])
        adapter_score = score_prediction(record["source_email"], adapter_prediction, record["target_slack"])
        base_scores.append(base_score)
        adapter_scores.append(adapter_score)
        examples.append(
            {
                "id": record["id"],
                "source_email": record["source_email"],
                "reference": record["target_slack"],
                "base_prediction": base_prediction,
                "adapter_prediction": adapter_prediction,
                "base_scores": base_score,
                "adapter_scores": adapter_score,
            }
        )

    results = {
        "generated_at": run_id,
        "model_name": settings.model_name,
        "adapter_path": str(settings.adapter_path),
        "sample_size": len(test_records),
        "aggregate": {
            "base": summarize_scores(base_scores),
            "adapter": summarize_scores(adapter_scores),
        },
        "examples": examples,
    }
    _write_json(run_dir / "results.json", results)
    (run_dir / "report.md").write_text(_render_report(results), encoding="utf-8")
    return run_dir
