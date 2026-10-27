"""FastAPI 应用装配。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .augment import rag
from .config import settings
from .middleware import AuthMiddleware, RateLimitMiddleware
from .ollama_client import OllamaError, client
from .protocols import protocol_catalog
from .routers import admin, gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("gateway")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时把知识库索引建好，避免首个请求卡住
    try:
        rag.get_store().build(force=True)
        log.info("知识库索引就绪：%s", rag.get_store().stats())
    except Exception as exc:  # noqa: BLE001
        log.warning("知识库索引失败：%s", exc)
    try:
        names = await client.model_names()
        log.info("Ollama 已连接，模型：%s", ", ".join(names) or "（空）")
    except Exception as exc:  # noqa: BLE001
        log.warning("Ollama 未就绪：%s", exc)
    log.info("鉴权：%s | 限流：%s 次/分钟", "开启" if settings.auth_enabled else "关闭", settings.rate_limit_per_minute)
    yield
    log.info("服务关闭")


app = FastAPI(
    title="Ollama Gateway",
    description=(
        "把本地 Ollama 包装成可对外服务的 LLM 网关：\n\n"
        "- **四协议统一**：Ollama 原生 / OpenAI Chat Completions / OpenAI Responses / Anthropic Messages\n"
        "- **四类外挂**：工具调用、知识库检索(RAG)、会话记忆、输入护栏\n"
        "- **对外必备**：API Key 鉴权、滑动窗口限流、用量统计\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

app.include_router(gateway.router)
app.include_router(admin.router)


@app.exception_handler(OllamaError)
async def ollama_error_handler(request: Request, exc: OllamaError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"type": "upstream_error", "message": exc.message}},
    )


@app.get("/", tags=["运维"], summary="服务概览")
async def root() -> dict[str, Any]:
    return {
        "service": settings.service_name,
        "version": "1.0.0",
        "default_model": settings.default_model,
        "protocols": protocol_catalog(),
        "augmentations": list(__import__("app.augment", fromlist=["AVAILABLE"]).AVAILABLE),
        "docs": "/docs",
        "playground": "/ui",
        "health": "/healthz",
    }


@app.get("/healthz", tags=["运维"], summary="存活检查")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["运维"], summary="就绪检查（会真的连一次 Ollama）")
async def readyz() -> JSONResponse:
    try:
        names = await client.model_names()
        ok = bool(names)
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "status": "ready" if ok else "no_model",
                "ollama": settings.ollama_base_url,
                "models": names,
                "knowledge_base": rag.get_store().stats(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "ollama": settings.ollama_base_url, "error": str(exc)},
        )


if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")


@app.get("/playground", include_in_schema=False)
async def playground() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))
