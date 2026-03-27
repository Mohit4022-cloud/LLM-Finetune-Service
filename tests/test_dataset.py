from llm_finetune_service.config import Settings
from llm_finetune_service.data.dataset_builder import build_dataset
from llm_finetune_service.data.validation import validate_dataset_dir


def test_dataset_build_is_deterministic(tmp_path):
    settings = Settings()
    settings.dataset_dir = tmp_path / "splits"
    result_a = build_dataset(settings, output_dir=settings.dataset_dir)
    result_b = build_dataset(settings, output_dir=settings.dataset_dir)
    assert result_a.split_counts == result_b.split_counts
    assert result_a.records[0]["source_email"] == result_b.records[0]["source_email"]


def test_dataset_validation_passes_for_generated_data(tmp_path):
    settings = Settings()
    settings.dataset_dir = tmp_path / "splits"
    build_dataset(settings, output_dir=settings.dataset_dir)
    errors = validate_dataset_dir(settings.dataset_dir)
    assert errors == []
