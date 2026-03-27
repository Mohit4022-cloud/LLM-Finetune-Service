# Evaluation Report

This document summarizes one real local training + evaluation pass run from this repository.

## Run metadata

- Date: `2026-03-27`
- Python: `3.12.13`
- Platform: `macOS-26.2-arm64-arm-64bit`
- Device: `mps`
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Training artifact: `artifacts/train_runs/20260327-164952/adapter`
- Evaluation artifact: `artifacts/eval/20260327-165119/`

## Exact commands used

```bash
poetry env use /opt/homebrew/bin/python3.12
poetry install
poetry run python scripts/build_dataset.py
poetry run python scripts/validate_dataset.py
PYTORCH_ENABLE_MPS_FALLBACK=1 MAX_STEPS=3 TRAINING_EPOCHS=1 poetry run python scripts/train.py
PYTORCH_ENABLE_MPS_FALLBACK=1 ADAPTER_PATH=artifacts/train_runs/20260327-164952/adapter EVAL_LIMIT=10 poetry run python scripts/evaluate.py
```

## Training summary

- Training runtime: `21.95s`
- Train loss: `1.7319`
- Validation loss: `1.5290`
- LoRA config:
  - `r=16`
  - `alpha=32`
  - `dropout=0.05`
- Training budget for this proof run:
  - `max_steps=3`
  - `epochs=1`

This was intentionally a short verification run, not a quality-maximizing experiment.

## Held-out evaluation summary

The evaluation was run on the first `10` examples from the held-out `test.jsonl` split.

Aggregate heuristic scores:

- Base overall score: `0.5311`
- Adapter overall score: `0.6459`
- Base compression ratio: `1.2552`
- Adapter compression ratio: `0.8770`

Interpretation:

- The short adapter run improved the aggregate heuristic score on this bounded held-out slice.
- The adapter more often reduced overly formal, email-like continuations from the base model.
- Quality is still mixed, which is expected given the tiny training budget.

## Qualitative examples

### Vendor coordination

**Source**

> I am writing to inform you that we need confirmation from the vendor before finalizing the purchasing timeline. We should keep vendor informed on the payroll reporting refresh so there are no surprises around the Q3 planning packet. This is one of the remaining dependencies before the broader rollout. Please confirm next steps by tomorrow morning. Kind regards,

**Reference**

> Quick vendor note: we need vendor confirmation before locking the purchasing timeline. This affects the payroll reporting refresh and the current work on the Q3 planning packet. Can someone confirm next steps by tomorrow morning?

**Base**

> [Enterprise Name] Dear [Vendor Name], I am writing to inform you that we need confirmation from the vendor before finalizing the purchasing timeline...

**Adapter**

> Confirm vendor request for payroll reporting refresh. We'll keep vendor informed on Q3 planning packet. Next steps: tomorrow morning.

### Escalation

**Source**

> For the sake of clarity, a security finding needs review before we can proceed with deployment. The current status has implications for the product roadmap on the mobile checkout launch, especially around the billing reconciliation run. This is one of the remaining dependencies before the broader rollout. Please confirm next steps by next Tuesday. Kind regards,

**Base**

> Dear [Name], Thank you for your response to our security finding...

**Adapter**

> Can you confirm the next steps for the security finding that has implications for the mobile checkout launch? Next Tuesday is the deadline for this.

## Limitations of this evidence run

- The run used `max_steps=3`, so it demonstrates end-to-end capability, not final task quality.
- The eval sample was limited to `10` held-out examples to keep the local run tractable.
- Some adapter outputs still copy too much structure from the source or produce list-like artifacts.

The full machine-readable outputs are in `artifacts/eval/20260327-165119/results.json`.
