"""对外服务的入口：同一个内核，四种协议外壳。"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import augment
from ..config import settings
from ..middleware import UsageRecord, usage_store
from ..ollama_client import OllamaError
from ..protocols import get_adapter, protocol_catalog
from ..schemas import ChatRequest

router = APIRouter()


def _error_body(protocol: str, message: str, err_type: str = "invalid_request_error") -> dict[str, Any]:
    """错误体也按各家风格返回，客户端 SDK 才能正确解析。"""
    if protocol == "anthropic":
        return {"type": "error", "error": {"type": err_type, "message": message}}
    return {"error": {"type": err_type, "message": message, "code": 400}}


def _record(
    background: BackgroundTasks,
    request: Request,
    protocol: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    background.add_task(
        usage_store.record,
        UsageRecord(
            ts=time.time(),
            path=request.url.path,
            protocol=protocol,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            status=status,
            api_key=getattr(request.state, "api_key", "anonymous"),
            extra=extra or {},
        ),
    )


async def _stream_with_fallback(req: ChatRequest, adapter) -> AsyncIterator[str]:
    """流式输出；任何异常都以该协议的 error 帧收尾，而不是直接断连接。"""
    try:
        async for frame in adapter.stream_frames(augment.run_stream(req), req.model):
            yield frame
    except OllamaError as exc:
        yield adapter.sse(_error_body(adapter.name, exc.message), event="error") if adapter.name != "ollama" else adapter.ndjson({"error": exc.message})
    except Exception as exc:  # noqa: BLE001
        yield adapter.ndjson({"error": f"服务内部错误：{exc}"}) if adapter.name == "ollama" else adapter.sse(
            _error_body(adapter.name, f"服务内部错误：{exc}", "server_error"), event="error"
        )


def _make_handler(protocol: str):
    adapter = get_adapter(protocol)

    async def handler(request: Request, background: BackgroundTasks):
        started = time.perf_counter()
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                status_code=400,
                content=_error_body(protocol, "请求体不是合法 JSON"),
            )
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content=_error_body(protocol, "请求体必须是 JSON 对象"))

        # 1. 协议 -> 内部格式
        try:
            req = adapter.parse_request(body, dict(request.headers))
        except ValueError as exc:
            return JSONResponse(status_code=400, content=_error_body(protocol, str(exc)))

        # 2. 模型必须存在（提前报错，流式时也能给出正确的 HTTP 状态码）
        try:
            await _client().ensure_model(req.model)
        except OllamaError as exc:
            _record(background, request, protocol, req.model, 0, 0, 0.0, f"error:{exc.status_code}")
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_body(protocol, exc.message, "model_not_found" if exc.status_code == 400 else "upstream_error"),
            )

        # 3. 流式
        if req.stream:
            _record(
                background,
                request,
                protocol,
                req.model,
                0,
                0,
                round((time.perf_counter() - started) * 1000, 1),
                "stream",
                {"augmentations": req.augmentations},
            )
            return StreamingResponse(
                _stream_with_fallback(req, adapter),
                media_type=adapter.stream_media_type,
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 4. 非流式
        try:
            resp, diag = await augment.run(req)
        except OllamaError as exc:
            _record(background, request, protocol, req.model, 0, 0, 0.0, f"error:{exc.status_code}")
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_body(protocol, exc.message, "upstream_error"),
            )

        out = adapter.render_response(resp)
        _record(
            background,
            request,
            protocol,
            resp.model,
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            resp.latency_ms,
            "ok",
            {"augmentations": resp.augmentations_used, "tool_rounds": len(resp.tool_trace)},
        )
        return JSONResponse(content=out)

    handler.__name__ = f"chat_{protocol.replace('-', '_')}"
    return handler


def _client():
    from ..ollama_client import client

    return client


for _proto in ("ollama", "openai-chat", "openai-responses", "anthropic"):
    _adapter = get_adapter(_proto)
    router.add_api_route(
        _adapter.path,
        _make_handler(_proto),
        methods=["POST"],
        name=f"chat_{_proto}",
        summary=f"{_proto} 协议对话（{_adapter.path}）",
        tags=[f"协议 / {_proto}"],
    )


@router.get("/v1/meta/protocols", tags=["协议 / 元信息"], summary="列出支持的协议")
async def list_protocols() -> dict[str, Any]:
    return {
        "service": settings.service_name,
        "default_model": settings.default_model,
        "protocols": protocol_catalog(),
    }


@router.get("/v1/meta/augmentations", tags=["协议 / 元信息"], summary="列出可用的外挂")
async def list_augmentations() -> dict[str, Any]:
    from ..augment import rag, tools

    return {
        "available": list(augment.AVAILABLE),
        "default": list(augment.DEFAULT),
        "usage": "在任意协议的请求体里加 augmentations 字段，例如 [\"rag\",\"tools\"]",
        "tools": [{"name": k, "description": v["description"]} for k, v in tools.TOOL_REGISTRY.items()],
        "knowledge_base": rag.get_store().stats(),
    }
