from __future__ import annotations

from typing import Optional


class CacheClient:
    """Redis cache with an in-memory fallback."""

    def __init__(self, host: str, port: int, db: int):
        self.redis_client = None
        self.memory_cache: dict[str, str] = {}
        try:
            import redis

            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                socket_connect_timeout=1,
                decode_responses=True,
            )
            client.ping()
            self.redis_client = client
        except Exception:
            self.redis_client = None

    @property
    def backend(self) -> str:
        return "redis" if self.redis_client is not None else "memory"

    def get(self, key: str) -> Optional[str]:
        if self.redis_client is not None:
            try:
                return self.redis_client.get(key)
            except Exception:
                return None
        return self.memory_cache.get(key)

    def set(self, key: str, value: str, ttl: int) -> None:
        if self.redis_client is not None:
            try:
                self.redis_client.setex(key, ttl, value)
                return
            except Exception:
                pass
        self.memory_cache[key] = value
