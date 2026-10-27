"""Ollama 上游客户端。

只做一件事：把内部 ``ChatRequest`` 发给 Ollama 的 ``/api/chat``，把结果解析成
内部 ``ChatResponse``。所有协议适配器最终都落到这里。

之所以不直接用官方 ``ollama`` Python SDK：
1. 需要精确控制流式解析（对外要做 SSE，边界行为必须自己掌握）；
2. 需要拿到 token 计数、耗时等原始字段来做对比实验；
3. 少一个依赖，部署更轻。
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx

from .config import settings
from .schemas import ChatRequest, ChatResponse, Message, ToolCall, Usage


class OllamaError(RuntimeError):
    """上游调用失败。"""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelNotFoundError(OllamaError):
    def __init__(self, model: str, available: list[str] | None = None):
        hint = ""
        if available:
            hint = f"；当前已有模型：{', '.join(available)}"
        super().__init__(
            f"模型 {model} 未安装。先执行 `ollama pull {model}`{hint}", status_code=400
        )


def _build_payload(req: ChatRequest, stream: bool) -> dict[str, Any]:
    """内部请求 -> Ollama /api/chat 请求体。"""
    options: dict[str, Any] = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.top_p is not None:
        options["top_p"] = req.top_p
    if req.max_tokens is not None:
        # Ollama 里叫 num_predict
        options["num_predict"] = req.max_tokens
    if req.stop:
        options["stop"] = req.stop

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        "stream": stream,
    }
    if options:
        payload["options"] = options
    if req.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in req.tools
        ]
    if req.json_schema:
        # Ollama 支持传 JSON Schema 做约束解码
        payload["format"] = req.json_schema
    return payload


def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall] | None:
    if not raw:
        return None
    calls: list[ToolCall] = []
    for item in raw:
        fn = item.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append(
            ToolCall(
                id=item.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                name=fn.get("name", ""),
                arguments=args or {},
            )
        )
    return calls


def _finish_reason(done_reason: str | None, has_tools: bool) -> str:
    if has_tools:
        return "tool_calls"
    if done_reason == "length":
        return "length"
    return "stop"


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    # ------------------------------------------------------------------ 基础
    async def _get(self, path: str) -> dict[str, Any]:
        try:
            async with self._client() as c:
                r = await c.get(path)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"连不上 Ollama（{self.base_url}）。请先启动 `ollama serve`。", 503
            ) from exc
        if r.status_code >= 400:
            raise OllamaError(f"Ollama 返回 {r.status_code}: {r.text[:300]}", r.status_code)
        return r.json()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._client() as c:
                r = await c.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"连不上 Ollama（{self.base_url}）。请先启动 `ollama serve`。", 503
            ) from exc
        except httpx.ReadTimeout as exc:
            raise OllamaError("Ollama 响应超时，可尝试调小 max_tokens 或换更小的模型。", 504) from exc
        if r.status_code >= 400:
            raise OllamaError(f"Ollama 返回 {r.status_code}: {r.text[:300]}", r.status_code)
        return r.json()

    # ------------------------------------------------------------------ 模型管理
    async def list_models(self) -> list[dict[str, Any]]:
        data = await self._get("/api/tags")
        models = data.get("models", [])
        for m in models:
            details = m.get("details") or {}
            m["size_gb"] = round((m.get("size") or 0) / 1e9, 2)
            m["parameter_size"] = details.get("parameter_size")
            m["quantization"] = details.get("quantization_level")
        return models

    async def model_names(self) -> list[str]:
        return [m["name"] for m in await self.list_models()]

    async def show_model(self, model: str) -> dict[str, Any]:
        return await self._post("/api/show", {"model": model})

    async def delete_model(self, model: str) -> dict[str, Any]:
        try:
            async with self._client() as c:
                r = await c.request("DELETE", "/api/delete", json={"model": model})
        except httpx.ConnectError as exc:
            raise OllamaError(f"连不上 Ollama（{self.base_url}）。", 503) from exc
        if r.status_code >= 400:
            raise OllamaError(f"删除失败 {r.status_code}: {r.text[:200]}", r.status_code)
        return {"deleted": model}

    async def pull_model(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """流式拉取模型，逐条 yield 进度事件。"""
        payload = {"model": model, "stream": True}
        try:
            async with self._client() as c:
                async with c.stream("POST", "/api/pull", json=payload) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode(errors="ignore")
                        raise OllamaError(f"拉取失败 {r.status_code}: {body[:300]}", r.status_code)
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        total = evt.get("total")
                        completed = evt.get("completed")
                        if total:
                            evt["percent"] = round(completed / total * 100, 1)
                        yield evt
        except httpx.ConnectError as exc:
            raise OllamaError(f"连不上 Ollama（{self.base_url}）。", 503) from exc

    # ------------------------------------------------------------------ 推理
    async def chat(self, req: ChatRequest) -> ChatResponse:
        """非流式对话。"""
        started = time.perf_counter()
        data = await self._post("/api/chat", _build_payload(req, stream=False))
        latency = (time.perf_counter() - started) * 1000

        msg = data.get("message") or {}
        tool_calls = _parse_tool_calls(msg.get("tool_calls"))
        message = Message(
            role="assistant",
            content=msg.get("content") or "",
            tool_calls=tool_calls,
        )
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count") or 0,
            completion_tokens=data.get("eval_count") or 0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return ChatResponse(
            model=req.model,
            message=message,
            finish_reason=_finish_reason(data.get("done_reason"), bool(tool_calls)),
            usage=usage,
            latency_ms=round(latency, 1),
            upstream_model=data.get("model"),
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """流式对话，yield 内部事件：

        - ``{"type": "delta", "text": "..."}``
        - ``{"type": "tool_calls", "calls": [...]}``
        - ``{"type": "done", "finish_reason": "...", "usage": {...}, "latency_ms": n}``
        """
        started = time.perf_counter()
        payload = _build_payload(req, stream=True)
        collected_calls: list[dict[str, Any]] = []
        prompt_tokens = completion_tokens = 0
        done_reason: str | None = None

        try:
            async with self._client() as c:
                async with c.stream("POST", "/api/chat", json=payload) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode(errors="ignore")
                        raise OllamaError(f"Ollama 返回 {r.status_code}: {body[:300]}", r.status_code)
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("error"):
                            raise OllamaError(str(evt["error"]))
                        msg = evt.get("message") or {}
                        if msg.get("content"):
                            yield {"type": "delta", "text": msg["content"]}
                        if msg.get("tool_calls"):
                            collected_calls.extend(msg["tool_calls"])
                        if evt.get("done"):
                            prompt_tokens = evt.get("prompt_eval_count") or 0
                            completion_tokens = evt.get("eval_count") or 0
                            done_reason = evt.get("done_reason")
        except httpx.ConnectError as exc:
            raise OllamaError(f"连不上 Ollama（{self.base_url}）。", 503) from exc

        calls = _parse_tool_calls(collected_calls)
        if calls:
            yield {"type": "tool_calls", "calls": [c.model_dump() for c in calls]}

        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        usage.total_tokens = prompt_tokens + completion_tokens
        yield {
            "type": "done",
            "finish_reason": _finish_reason(done_reason, bool(calls)),
            "usage": usage.model_dump(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    async def ensure_model(self, model: str) -> None:
        names = await self.model_names()
        # Ollama 允许省略 tag，默认 latest
        if model in names:
            return
        if f"{model}:latest" in names:
            return
        raise ModelNotFoundError(model, names)


client = OllamaClient()
