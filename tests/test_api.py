from fastapi.testclient import TestClient

from llm_finetune_service.api.app import create_app
from llm_finetune_service.config import Settings


def _build_client():
    settings = Settings()
    settings.inference_mode = "dev_fallback"
    settings.allow_dev_fallback = True
    app = create_app(settings)
    return TestClient(app)


def test_health_contract():
    with _build_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "dev_fallback"
        assert "cache_backend" in payload


def test_generate_contract():
    with _build_client() as client:
        response = client.post("/generate", json={"text": "I am writing to inform you that the launch is delayed."})
        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "dev_fallback"
        assert payload["cached"] is False
        assert payload["generated_text"]
