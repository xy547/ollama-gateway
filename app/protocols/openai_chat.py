"""OpenAI Chat Completions：``POST /v1/chat/completions``。

这是目前事实上的行业标准，绝大多数客户端 SDK（openai-python、LangChain、Dify、
Cherry Studio、各类"自定义 API 地址"的聊天客户端）默认说的都是这套话。

特点（对比 Ollama 原生）：
- 扁平参数：``temperature`` / ``max_tokens`` 直接在顶层，不再包一层 ``options``；
- 返回值有 ``id`` / ``object`` / ``created`` 三件套，把"回答"包在 ``choices[]`` 数组里；
- ``finish_reason`` 取值是 ``stop`` / ``length`` / ``tool_calls``（Ollama 是 ``done_reason: stop|length``）；
- 流式是 SSE，且以一个特殊的 ``data: [DONE]`` 结束；
- 工具调用的 ``arguments`` 是 **JSON 字符串**而不是对象（这里最容易踩坑）。
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..config import settings
from ..schemas import ChatRequest, ChatResponse, Message, ToolCall, ToolSpec
from .base import ProtocolAdapter, new_id, now_ts


def _parse_content(content: Any) -> str:
    """兼容纯字符串与 OpenAI 的多段 content 数组。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class OpenAIChatAdapter(ProtocolAdapter):
    name = "openai-chat"
    path = "/v1/chat/completions"
    auth_hint = "Authorization: Bearer <key>"

    def parse_request(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> ChatRequest:
        if not body.get("messages"):
            raise ValueError("messages 不能为空")
        tools = None
        if body.get("tools"):
            tools = []
            for t in body["tools"]:
                fn = t.get("function") or t
                tools.append(
                    ToolSpec(
                        name=fn.get("name", ""),
                        description=fn.get("description", ""),
                        parameters=fn.get("parameters") or {},
                    )
                )

        messages: list[Message] = []
        for m in body["messages"]:
            role = m.get("role", "user")
            tcs = None
            if m.get("tool_calls"):
                tcs = []
                for i, c in enumerate(m["tool_calls"]):
                    fn = c.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    tcs.append(
                        ToolCall(id=c.get("id") or f"call_{i}", name=fn.get("name", ""), arguments=args or {})
                    )
            messages.append(
                Message(
                    role=role,
                    content=_parse_content(m.get("content")),
                    tool_calls=tcs,
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                )
            )

        # response_format: {"type":"json_schema","json_schema":{"schema":{...}}}
        schema = None
        rf = body.get("response_format") or {}
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            schema = ((rf.get("json_schema") or {}).get("schema")) or None

        return ChatRequest(
            model=body.get("model") or settings.default_model,
            messages=messages,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
            stop=body.get("stop") if isinstance(body.get("stop"), list) else None,
            stream=bool(body.get("stream")),
            tools=tools,
            json_schema=schema,
            augmentations=body.get("augmentations") or [],
            session_id=body.get("user") or body.get("session_id"),
        )

    def render_response(self, resp: ChatResponse, request_id: str | None = None) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": resp.message.content}
        if resp.message.tool_calls:
            msg["content"] = resp.message.content or None
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in resp.message.tool_calls
            ]
        return {
            "id": request_id or new_id("chatcmpl-"),
            "object": "chat.completion",
            "created": now_ts(),
            "model": resp.model,
            "choices": [{"index": 0, "message": msg, "finish_reason": resp.finish_reason}],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            "system_fingerprint": "ollama-gateway",
        }

    async def stream_frames(
        self,
        events: AsyncIterator[dict[str, Any]],
        model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        cid = request_id or new_id("chatcmpl-")
        created = now_ts()

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        yield self.sse(chunk({"role": "assistant", "content": ""}))
        async for evt in events:
            if evt["type"] == "delta":
                yield self.sse(chunk({"content": evt["text"]}))
            elif evt["type"] == "tool_calls":
                yield self.sse(
                    chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": c["id"],
                                    "type": "function",
                                    "function": {
                                        "name": c["name"],
                                        "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                                    },
                                }
                                for i, c in enumerate(evt["calls"])
                            ]
                        }
                    )
                )
            elif evt["type"] == "done":
                final = chunk({}, evt["finish_reason"])
                final["usage"] = evt["usage"]
                yield self.sse(final)
        yield "data: [DONE]\n\n"
