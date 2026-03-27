# Reproducibility Guide

## Supported versions

- Python `3.11` or `3.12`
- Poetry `2.x`

Python `3.14` is not supported for this repo. The original workspace showed a prior compatibility failure under `3.14.3`, and this project now documents the supported path explicitly instead of hiding that failure behind mock artifacts.

## Environment setup

```bash
poetry env use 3.11
poetry install
```

## Regenerate the dataset

```bash
make data
make validate-data
make data-preview
```

## Run LoRA fine-tuning

```bash
make train
```

Expected output:

- `artifacts/train_runs/<run_id>/adapter/`
- `artifacts/train_runs/<run_id>/train_config.json`
- `artifacts/train_runs/<run_id>/metrics.json`
- `artifacts/train_runs/<run_id>/environment.json`

## Run evaluation

```bash
make eval
```

Expected output:

- `artifacts/eval/<run_id>/results.json`
- `artifacts/eval/<run_id>/report.md`

## Serve the model

Base model only:

```bash
make serve-base
```

Adapter mode:

```bash
make serve-adapter
```

Development-only fallback mode:

```bash
make serve-dev
```

## Known blocker in this workspace

The current workspace interpreter is `Python 3.14.3`, and model-training dependencies are not installed under that interpreter. Because this repo now targets `>=3.11,<3.13`, a real training run still needs to happen in a supported local environment. That blocker is environmental, not hidden by code.
