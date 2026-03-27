from pathlib import Path

import pytest

from llm_finetune_service.config import Settings
from llm_finetune_service.inference.model import TextGenerator


def test_dev_fallback_requires_opt_in():
    settings = Settings()
    settings.inference_mode = "dev_fallback"
    settings.allow_dev_fallback = False
    with pytest.raises(RuntimeError):
        TextGenerator(settings)


def test_adapter_mode_requires_adapter_path():
    settings = Settings()
    settings.inference_mode = "adapter"
    settings.adapter_path = Path("missing-adapter")
    with pytest.raises(RuntimeError):
        TextGenerator(settings)
