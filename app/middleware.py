"""对外服务的三件套：鉴权、限流、用量统计。

Ollama 本身没有鉴权，直接 ``OLLAMA_HOST=0.0.0.0`` 暴露到公网 = 谁都能白嫖你的
显卡。所以把它包成对外服务时，这三件事必须要做，否则不是"服务"而是"漏洞"。
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

PUBLIC_PATHS = {"/", "/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json", "/stats"}

#: 只有"真的要跑模型"的路径才计数限流。
#: 管理面（查模型列表、看用量、写知识库）不烧显卡，没必要卡用户。
RATE_LIMITED_PATHS = {
    "/api/chat",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
    "/api/compare",
}


# ------------------------------------------------------------------ 鉴权
def extract_key(request: Request) -> str | None:
    """支持三种常见的 key 传法。"""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    for header in ("x-api-key", "api-key"):
        val = request.headers.get(header)
        if val:
            return val.strip()
    return request.query_params.get("api_key")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not settings.auth_enabled or path in PUBLIC_PATHS or path.startswith("/ui"):
            request.state.api_key = "anonymous"
            return await call_next(request)

        key = extract_key(request)
        if not key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "缺少 API Key。请通过 Authorization: Bearer <key> 或 x-api-key 请求头传入。",
                    }
                },
            )
        if key not in settings.key_list:
            return JSONResponse(
                status_code=403,
                content={"error": {"type": "permission_error", "message": "API Key 无效。"}},
            )
        request.state.api_key = key[:6] + "***"
        return await call_next(request)


# ------------------------------------------------------------------ 限流
class SlidingWindowLimiter:
    """滑动窗口限流：每个 key 每分钟最多 N 次。"""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        """返回 (是否放行, 剩余次数, 需等待秒数)。"""
        if self.per_minute <= 0:
            return True, -1, 0.0
        now = time.time()
        window = 60.0
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= self.per_minute:
                retry = window - (now - q[0])
                return False, 0, round(max(retry, 0.1), 1)
            q.append(now)
            return True, self.per_minute - len(q), 0.0

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key:
                self._hits.pop(key, None)
            else:
                self._hits.clear()


limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path not in RATE_LIMITED_PATHS:
            return await call_next(request)
        key = getattr(request.state, "api_key", "anonymous")
        ok, remaining, retry = limiter.check(key)
        if not ok:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(retry) + 1)},
                content={
                    "error": {
                        "type": "rate_limit_error",
                        "message": f"超过限流阈值（{limiter.per_minute} 次/分钟），请 {retry}s 后重试。",
                    }
                },
            )
        response = await call_next(request)
        if remaining >= 0:
            response.headers["X-RateLimit-Limit"] = str(limiter.per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ------------------------------------------------------------------ 用量
@dataclass
class UsageRecord:
    ts: float
    path: str
    protocol: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: str = "ok"
    api_key: str = "anonymous"
    extra: dict[str, Any] = field(default_factory=dict)


class UsageStore:
    """内存级用量统计（重启即清零，够演示用；生产应落 TSDB）。"""

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self._records: deque[UsageRecord] = deque(maxlen=capacity)
        self._lock = Lock()
        self._totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        )

    def record(self, rec: UsageRecord) -> None:
        with self._lock:
            self._records.append(rec)
            t = self._totals[rec.protocol]
            t["requests"] += 1
            t["prompt_tokens"] += rec.prompt_tokens
            t["completion_tokens"] += rec.completion_tokens
            t["latency_ms"] += rec.latency_ms

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_protocol = {
                k: {
                    "requests": int(v["requests"]),
                    "prompt_tokens": int(v["prompt_tokens"]),
                    "completion_tokens": int(v["completion_tokens"]),
                    "avg_latency_ms": round(v["latency_ms"] / v["requests"], 1) if v["requests"] else 0,
                }
                for k, v in self._totals.items()
            }
            recent = [asdict(r) for r in list(self._records)[-50:]]
            total_requests = sum(v["requests"] for v in self._totals.values())
        return {
            "total_requests": int(total_requests),
            "by_protocol": by_protocol,
            "recent": recent,
        }

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._totals.clear()


usage_store = UsageStore()
