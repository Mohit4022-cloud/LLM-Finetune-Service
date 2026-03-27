#!/usr/bin/env python3
from llm_finetune_service.config import Settings
from llm_finetune_service.eval.run_eval import run_evaluation


if __name__ == "__main__":
    run_dir = run_evaluation(Settings())
    print(f"Evaluation complete. Results saved to {run_dir}")
