"""管理面：模型下载 / 删除、知识库、会话、用量、四协议对比。"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import augment
from ..augment import memory, rag
from ..config import settings
from ..middleware import usage_store
from ..ollama_client import OllamaError, client
from ..protocols import ADAPTERS, get_adapter

router = APIRouter()


# ============================================================== 模型管理
@router.get("/api/models", tags=["管理 / 模型"], summary="列出已安装模型")
async def list_models() -> dict[str, Any]:
    try:
        models = await client.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return {
        "default": settings.default_model,
        "count": len(models),
        "models": [
            {
                "name": m["name"],
                "size_gb": m.get("size_gb"),
                "parameter_size": m.get("parameter_size"),
                "quantization": m.get("quantization"),
                "capabilities": m.get("capabilities", []),
                "modified_at": m.get("modified_at"),
            }
            for m in models
        ],
    }


class PullBody(BaseModel):
    model: str = Field(..., description="模型名，如 qwen2.5:1.5b")


@router.post("/api/models/pull", tags=["管理 / 模型"], summary="下载模型（SSE 进度）")
async def pull_model(body: PullBody) -> StreamingResponse:
    """拉模型。走 Ollama 的 /api/pull，逐条把进度推给前端。"""

    async def gen() -> AsyncIterator[str]:
        try:
            async for evt in client.pull_model(body.model):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except OllamaError as exc:
            yield f"data: {json.dumps({'error': exc.message}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/api/models/{name:path}/show", tags=["管理 / 模型"], summary="查看模型详情")
async def show_model(name: str) -> dict[str, Any]:
    try:
        info = await client.show_model(name)
    except OllamaError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return {
        "model": name,
        "capabilities": info.get("capabilities", []),
        "details": info.get("details", {}),
        "parameters": info.get("parameters"),
        "template_preview": (info.get("template") or "")[:600],
    }


@router.delete("/api/models/{name:path}", tags=["管理 / 模型"], summary="删除模型")
async def delete_model(name: str) -> dict[str, Any]:
    try:
        return await client.delete_model(name)
    except OllamaError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


# ============================================================== 用量
@router.get("/stats", tags=["管理 / 观测"], summary="用量统计")
async def stats() -> dict[str, Any]:
    data = usage_store.summary()
    data["service"] = settings.service_name
    data["auth_enabled"] = settings.auth_enabled
    data["rate_limit_per_minute"] = settings.rate_limit_per_minute
    return data


@router.post("/stats/reset", tags=["管理 / 观测"], summary="清空用量")
async def reset_stats() -> dict[str, Any]:
    usage_store.clear()
    return {"ok": True}


# ============================================================== 知识库
class DocBody(BaseModel):
    filename: str = Field(..., description="文件名，建议 .md 或 .txt")
    content: str = Field(..., description="正文")


@router.get("/api/knowledge", tags=["管理 / 知识库"], summary="知识库统计")
async def knowledge_stats() -> dict[str, Any]:
    return rag.get_store().stats()


@router.post("/api/knowledge", tags=["管理 / 知识库"], summary="新增/覆盖一篇文档")
async def add_doc(body: DocBody) -> dict[str, Any]:
    return rag.get_store().add_document(body.filename, body.content)


# ============================================================== 会话
@router.get("/api/sessions", tags=["管理 / 记忆"], summary="会话列表")
async def sessions() -> dict[str, Any]:
    return {"sessions": memory.list_sessions()}


@router.delete("/api/sessions/{session_id}", tags=["管理 / 记忆"], summary="清空某个会话")
async def clear_session(session_id: str) -> dict[str, Any]:
    return memory.clear_session(session_id)


# ============================================================== 四协议对比
class CompareBody(BaseModel):
    prompt: str = Field(..., description="要问的话")
    model: str | None = None
    system: str = "你是一个简洁的助手，回答控制在 80 字以内。"
    max_tokens: int = 256
    augmentations: list[str] = Field(default_factory=list, description="外挂，空数组=裸模型")
    protocols: list[str] = Field(default_factory=lambda: list(ADAPTERS))


def _build_body(protocol: str, prompt: str, model: str, system: str, max_tokens: int, augs: list[str]) -> dict[str, Any]:
    """按各协议的写法，手工构造"同一个请求"。这部分本身就是一份协议对照表。"""
    base_aug = {"augmentations": augs}
    if protocol == "ollama":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
            **base_aug,
        }
    if protocol == "openai-chat":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
            **base_aug,
        }
    if protocol == "openai-responses":
        return {
            "model": model,
            "instructions": system,  # 独立字段，不在 input 里
            "input": prompt,  # 可以是字符串
            "max_output_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
            **base_aug,
        }
    if protocol == "anthropic":
        return {
            "model": model,
            "system": system,  # 顶层字段
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,  # 必填
            "temperature": 0.3,
            "stream": False,
            **base_aug,
        }
    raise ValueError(protocol)


def _extract_text(protocol: str, raw: dict[str, Any]) -> str:
    """从各自的响应结构里把文本捞出来——四种路径，正好体现了结构差异。"""
    try:
        if protocol == "ollama":
            return raw["message"]["content"]
        if protocol == "openai-chat":
            return raw["choices"][0]["message"]["content"] or ""
        if protocol == "openai-responses":
            for item in raw.get("output", []):
                if item.get("type") == "message":
                    return "".join(p.get("text", "") for p in item.get("content", []))
            return ""
        if protocol == "anthropic":
            return "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
    except (KeyError, IndexError, TypeError):
        return ""
    return ""


def _finish_of(protocol: str, raw: dict[str, Any]) -> str:
    if protocol == "ollama":
        return raw.get("done_reason") or "stop"
    if protocol == "openai-chat":
        return raw["choices"][0].get("finish_reason") or "stop"
    if protocol == "openai-responses":
        return raw.get("status") or "completed"
    if protocol == "anthropic":
        return raw.get("stop_reason") or "end_turn"
    return ""


def _usage_of(protocol: str, raw: dict[str, Any]) -> dict[str, int]:
    if protocol == "ollama":
        return {"prompt": raw.get("prompt_eval_count", 0), "completion": raw.get("eval_count", 0)}
    if protocol == "openai-chat":
        u = raw.get("usage") or {}
        return {"prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0)}
    if protocol == "openai-responses":
        u = raw.get("usage") or {}
        return {"prompt": u.get("input_tokens", 0), "completion": u.get("output_tokens", 0)}
    if protocol == "anthropic":
        u = raw.get("usage") or {}
        return {"prompt": u.get("input_tokens", 0), "completion": u.get("output_tokens", 0)}
    return {"prompt": 0, "completion": 0}


@router.post("/api/compare", tags=["管理 / 对比"], summary="同一句话走四种协议，返回字段对照")
async def compare(body: CompareBody) -> dict[str, Any]:
    model = body.model or settings.default_model
    try:
        await client.ensure_model(model)
    except OllamaError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    results: list[dict[str, Any]] = []
    for proto in body.protocols:
        if proto not in ADAPTERS:
            continue
        adapter = get_adapter(proto)
        raw_body = _build_body(proto, body.prompt, model, body.system, body.max_tokens, body.augmentations)
        started = time.perf_counter()
        try:
            req = adapter.parse_request(raw_body)
            resp, diag = await augment.run(req)
            out = adapter.render_response(resp)
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            results.append(
                {
                    "protocol": proto,
                    "path": adapter.path,
                    "request_sent": raw_body,
                    "response": out,
                    "text": _extract_text(proto, out),
                    "finish": _finish_of(proto, out),
                    "usage": _usage_of(proto, out),
                    "latency_ms": elapsed,
                    "ok": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "protocol": proto,
                    "path": adapter.path,
                    "request_sent": raw_body,
                    "error": str(exc),
                    "ok": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )

    return {
        "prompt": body.prompt,
        "model": model,
        "results": results,
        "cheatsheet": CHEATSHEET,
    }


CHEATSHEET = [
    ("端点", ["/api/chat", "/v1/chat/completions", "/v1/responses", "/v1/messages"]),
    ("输入字段", ["messages", "messages", "input", "messages"]),
    ("系统提示", ["messages[role=system]", "messages[role=system]", "instructions", "system（顶层）"]),
    ("输出长度", ["options.num_predict", "max_tokens", "max_output_tokens", "max_tokens（必填）"]),
    ("文本位置", ["message.content", "choices[0].message.content", "output[].content[].text", "content[].text"]),
    ("结束原因", ["done_reason", "finish_reason", "status", "stop_reason"]),
    ("结束取值", ["stop / length", "stop / length / tool_calls", "completed / incomplete", "end_turn / max_tokens / tool_use"]),
    ("token 字段", ["prompt_eval_count / eval_count", "usage.prompt_tokens", "usage.input_tokens", "usage.input_tokens"]),
    ("工具定义", ["tools[].function.parameters", "tools[].function.parameters", "tools[].parameters", "tools[].input_schema"]),
    ("工具调用", ["message.tool_calls", "message.tool_calls[].function", "output[] type=function_call", "content[] type=tool_use"]),
    ("流式格式", ["NDJSON", "SSE + [DONE]", "SSE（带 event 类型）", "SSE（带 event 类型）"]),
    ("鉴权头", ["无", "Authorization: Bearer", "Authorization: Bearer", "x-api-key + anthropic-version"]),
]
