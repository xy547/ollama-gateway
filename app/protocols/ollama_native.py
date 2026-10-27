"""Ollama 原生协议：``POST /api/chat``。

特点：
- 请求体最"薄"，参数放在 ``options`` 嵌套字典里（``num_predict`` 而不是 ``max_tokens``）；
- 流式是 **NDJSON**（一行一个 JSON），不是主流大厂用的 SSE；
- token 统计字段叫 ``prompt_eval_count`` / ``eval_count``，还额外给纳秒级耗时；
- 没有鉴权概念，靠网络边界隔离。
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

from ..config import settings
from ..schemas import ChatRequest, ChatResponse, Message, ToolCall, ToolSpec
from .base import ProtocolAdapter


class OllamaNativeAdapter(ProtocolAdapter):
    name = "ollama"
    path = "/api/chat"
    stream_media_type = "application/x-ndjson"
    auth_hint = "无"

    def parse_request(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> ChatRequest:
        if not body.get("messages"):
            raise ValueError("messages 不能为空")
        options = body.get("options") or {}
        tools = None
        if body.get("tools"):
            tools = [
                ToolSpec(
                    name=(t.get("function") or {}).get("name", ""),
                    description=(t.get("function") or {}).get("description", ""),
                    parameters=(t.get("function") or {}).get("parameters") or {},
                )
                for t in body["tools"]
            ]
        messages = []
        for m in body["messages"]:
            tcs = None
            if m.get("tool_calls"):
                tcs = [
                    ToolCall(
                        id=(c.get("id") or f"call_{i}"),
                        name=(c.get("function") or {}).get("name", ""),
                        arguments=(c.get("function") or {}).get("arguments") or {},
                    )
                    for i, c in enumerate(m["tool_calls"])
                ]
            messages.append(
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content") or "",
                    tool_calls=tcs,
                    tool_call_id=m.get("tool_call_id"),
                )
            )
        return ChatRequest(
            model=body.get("model") or settings.default_model,
            messages=messages,
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
            max_tokens=options.get("num_predict"),
            stop=options.get("stop"),
            stream=bool(body.get("stream")),
            tools=tools,
            json_schema=body.get("format") if isinstance(body.get("format"), dict) else None,
            augmentations=body.get("augmentations") or [],
            session_id=body.get("session_id"),
        )

    def render_response(self, resp: ChatResponse, request_id: str | None = None) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": resp.message.content}
        if resp.message.tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in resp.message.tool_calls
            ]
        return {
            "model": resp.upstream_model or resp.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
            "message": msg,
            "done": True,
            "done_reason": "length" if resp.finish_reason == "length" else "stop",
            "total_duration": int(resp.latency_ms * 1_000_000),
            "prompt_eval_count": resp.usage.prompt_tokens,
            "eval_count": resp.usage.completion_tokens,
        }

    async def stream_frames(
        self,
        events: AsyncIterator[dict[str, Any]],
        model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        created = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())
        async for evt in events:
            if evt["type"] == "delta":
                yield self.ndjson(
                    {
                        "model": model,
                        "created_at": created,
                        "message": {"role": "assistant", "content": evt["text"]},
                        "done": False,
                    }
                )
            elif evt["type"] == "tool_calls":
                yield self.ndjson(
                    {
                        "model": model,
                        "created_at": created,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": c["name"], "arguments": c["arguments"]}}
                                for c in evt["calls"]
                            ],
                        },
                        "done": False,
                    }
                )
            elif evt["type"] == "done":
                usage = evt["usage"]
                yield self.ndjson(
                    {
                        "model": model,
                        "created_at": created,
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "done_reason": "length" if evt["finish_reason"] == "length" else "stop",
                        "prompt_eval_count": usage["prompt_tokens"],
                        "eval_count": usage["completion_tokens"],
                    }
                )
