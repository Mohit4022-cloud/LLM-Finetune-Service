# Architecture Notes

## Goal

Demonstrate an end-to-end fine-tuning workflow for a narrow text transformation task:

`formal enterprise email -> concise Slack-style message`

## Components

### Dataset generation

`src/llm_finetune_service/data/dataset_builder.py`

- produces deterministic synthetic supervision
- writes train/validation/test splits
- preserves raw fields for auditability and reuse

### Dataset validation

`src/llm_finetune_service/data/validation.py`

- schema checks
- duplicate detection
- split leakage checks
- style sanity checks

### Training

`src/llm_finetune_service/training/train.py`

- loads split JSONL files through `datasets`
- renders instruction-style prompts
- masks prompt tokens in the label sequence
- fine-tunes LoRA adapters via `peft`
- stores run artifacts under `artifacts/train_runs/<run_id>/`

### Evaluation

`src/llm_finetune_service/eval/run_eval.py`

- loads the held-out test split
- generates outputs with the base model and the adapter model
- scores both using task-specific heuristics
- writes `results.json` and `report.md`

### Serving

`src/llm_finetune_service/api/app.py`

- FastAPI app
- explicit runtime modes
- optional Redis cache with in-memory fallback
- transparent health and generation metadata

## Why LoRA

LoRA is a good fit for this project because:

- the task is narrow and style-heavy
- parameter-efficient training keeps the project lightweight
- adapter artifacts are easier to store and swap than full model checkpoints
- the workflow is recognizable to recruiters and applied ML teams

## Artifact flow

1. `make data`
2. `make validate-data`
3. `make train`
4. `make eval`
5. `make serve-adapter`

Generated artifacts are intentionally separated from source code and excluded from git by default.
