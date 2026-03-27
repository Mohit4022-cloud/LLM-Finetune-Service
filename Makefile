.PHONY: help install data validate-data data-preview train eval serve-base serve-adapter serve-dev test clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies via Poetry"
	@echo "  make data           - Build deterministic train/validation/test splits"
	@echo "  make validate-data  - Run dataset validation checks"
	@echo "  make data-preview   - Print representative dataset samples"
	@echo "  make train          - Run LoRA fine-tuning"
	@echo "  make eval           - Compare base vs adapter outputs on the held-out split"
	@echo "  make serve-base     - Serve the base model only"
	@echo "  make serve-adapter  - Serve the base model plus a trained adapter"
	@echo "  make serve-dev      - Serve deterministic dev fallback mode"
	@echo "  make test           - Run the automated test suite"
	@echo "  make clean          - Remove generated artifacts"

install:
	poetry install

data:
	poetry run python scripts/build_dataset.py

validate-data:
	poetry run python scripts/validate_dataset.py

data-preview:
	poetry run python scripts/preview_dataset.py

train:
	poetry run python scripts/train.py

eval:
	poetry run python scripts/evaluate.py

serve-base:
	INFERENCE_MODE=base poetry run uvicorn llm_finetune_service.api.app:app --host 0.0.0.0 --port 8000

serve-adapter:
	INFERENCE_MODE=adapter poetry run uvicorn llm_finetune_service.api.app:app --host 0.0.0.0 --port 8000

serve-dev:
	INFERENCE_MODE=dev_fallback ALLOW_DEV_FALLBACK=true poetry run uvicorn llm_finetune_service.api.app:app --host 0.0.0.0 --port 8000

test:
	poetry run pytest

clean:
	rm -rf artifacts/train_runs artifacts/eval .pytest_cache .ruff_cache
