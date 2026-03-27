from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_finetune_service.config import Settings
from llm_finetune_service.inference.dev_fallback import generate_dev_fallback
from llm_finetune_service.training.prompts import render_prompt


@dataclass(slots=True)
class InferenceState:
    mode: str
    model_name: str
    adapter_loaded: bool
    adapter_path: str


class TextGenerator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.mode = self.settings.inference_mode
        self.model_name = self.settings.model_name
        self.adapter_path = str(self.settings.adapter_path)
        self.adapter_loaded = False
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        if self.mode == "dev_fallback":
            if not self.settings.allow_dev_fallback:
                raise RuntimeError("dev_fallback mode requires ALLOW_DEV_FALLBACK=true")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Model dependencies are not installed. Use Poetry with Python 3.11 or 3.12 and run 'poetry install'."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(self.settings.model_name, **model_kwargs)
        if torch.backends.mps.is_available() and "device_map" not in model_kwargs:
            base_model = base_model.to("mps")

        if self.mode == "adapter":
            adapter_path = Path(self.settings.adapter_path)
            if not adapter_path.exists():
                raise RuntimeError(f"Adapter mode requested but adapter path does not exist: {adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load LoRA adapters.") from exc
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.adapter_loaded = True
        elif self.mode == "base":
            self.model = base_model
        else:
            raise RuntimeError(f"Unsupported inference mode: {self.mode}")

        self.model.eval()

    def describe(self) -> InferenceState:
        return InferenceState(
            mode=self.mode,
            model_name=self.model_name,
            adapter_loaded=self.adapter_loaded,
            adapter_path=self.adapter_path,
        )

    def generate(self, text: str) -> str:
        if self.mode == "dev_fallback":
            return generate_dev_fallback(text)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        prompt_record = {
            "instruction": (
                "Rewrite the enterprise email as a concise, friendly Slack message. Preserve the key facts, "
                "requested action, and deadlines."
            ),
            "source_email": text,
            "target_slack": "",
        }
        prompt = render_prompt(prompt_record, include_target=False)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        )

        try:
            import torch

            if torch.cuda.is_available():
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            elif torch.backends.mps.is_available():
                inputs = {key: value.to("mps") for key, value in inputs.items()}
        except ImportError:
            pass

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=0.3,
            top_p=0.92,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Slack Message" in decoded:
            decoded = decoded.split("### Slack Message", 1)[1]
        return decoded.strip()
