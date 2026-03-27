from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from llm_finetune_service.config import Settings
from llm_finetune_service.inference.cache import CacheClient
from llm_finetune_service.inference.model import TextGenerator


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class GenerateResponse(BaseModel):
    generated_text: str
    mode: str
    model_name: str
    adapter_loaded: bool
    cached: bool
    latency_ms: int


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_id: str) -> None:
        now = time.time()
        timestamps = self._requests[client_id]
        while timestamps and now - timestamps[0] > self.window_seconds:
            timestamps.popleft()
        if len(timestamps) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
        timestamps.append(now)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="LLM Fine-Tune Service",
        description="Fine-tuned Slack-style rewriting service with explicit base, adapter, and dev fallback modes.",
        version="0.1.0",
    )

    app.state.settings = settings
    app.state.cache = CacheClient(settings.redis_host, settings.redis_port, settings.redis_db)
    app.state.generator = None
    app.state.rate_limiter = InMemoryRateLimiter(limit=10, window_seconds=60)

    @app.on_event("startup")
    async def startup() -> None:
        app.state.generator = TextGenerator(settings)

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "llm-finetune-service",
            "workflow": "dataset -> train -> evaluate -> serve",
            "docs": "/docs",
        }

    @app.get("/health")
    async def health() -> dict[str, object]:
        generator = app.state.generator
        if generator is None:
            raise HTTPException(status_code=503, detail="Generator is not initialized.")
        cache = app.state.cache
        state = generator.describe()
        return {
            "status": "healthy",
            "mode": state.mode,
            "base_model": state.model_name,
            "adapter_path": state.adapter_path,
            "adapter_loaded": state.adapter_loaded,
            "cache_backend": cache.backend,
            "rate_limit_enabled": True,
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: Request, payload: GenerateRequest) -> GenerateResponse:
        start = time.time()
        generator = app.state.generator
        if generator is None:
            raise HTTPException(status_code=503, detail="Generator is not initialized.")
        cache = app.state.cache
        client_id = request.client.host if request.client else "unknown"
        app.state.rate_limiter.check(client_id)
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        cache_key = f"rewrite:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        cached = cache.get(cache_key)
        state = generator.describe()
        if cached is not None:
            latency = int((time.time() - start) * 1000)
            return GenerateResponse(
                generated_text=cached,
                mode=state.mode,
                model_name=state.model_name,
                adapter_loaded=state.adapter_loaded,
                cached=True,
                latency_ms=latency,
            )

        try:
            generated = generator.generate(text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

        cache.set(cache_key, generated, ttl=settings.redis_ttl)
        latency = int((time.time() - start) * 1000)
        return GenerateResponse(
            generated_text=generated,
            mode=state.mode,
            model_name=state.model_name,
            adapter_loaded=state.adapter_loaded,
            cached=False,
            latency_ms=latency,
        )

    return app


app = create_app()
