"""Anthropic Messages：``POST /v1/messages``。

Claude 的接口是四家里"最不 OpenAI"的一个，差异主要在 **结构而不是命名**：

1. ``system`` 不是 messages 里的一条，而是**顶层独立字段**；
2. ``max_tokens`` **必填**（缺失直接 400），这是很多老客户端接 Claude 时第一个报错；
3. 响应 ``content`` 是 **block 数组**，文本是 ``content[0].text``；
4. 工具调用叫 ``tool_use`` block，工具回传叫 ``tool_result`` block，而且
   ``tool_result`` 要放进 **user** 消息里，不是单独的 ``role=tool``；
5. ``stop_reason`` 取值是 ``end_turn`` / ``max_tokens`` / ``tool_use`` / ``stop_sequence``；
6. 鉴权是 ``x-api-key`` + 必填的 ``anthropic-version`` 头（我们这里校验并透传语义）。
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from ..config import settings
from ..schemas import ChatRequest, ChatResponse, Message, ToolCall, ToolSpec
from .base import ProtocolAdapter, new_id

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "error": "end_turn",
}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


class AnthropicMessagesAdapter(ProtocolAdapter):
    name = "anthropic"
    path = "/v1/messages"
    auth_hint = "x-api-key: <key> + anthropic-version: 2023-06-01"

    def parse_request(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> ChatRequest:
        if not body.get("messages"):
            raise ValueError("messages 不能为空")
        if body.get("max_tokens") in (None, ""):
            # 真实 Anthropic 的硬性要求，故意保持一致
            raise ValueError("max_tokens: Field required（Anthropic 协议中该字段必填）")

        messages: list[Message] = []
        if body.get("system"):
            messages.append(Message(role="system", content=_text_of(body["system"])))

        for m in body["messages"]:
            role = m.get("role", "user")
            content = m.get("content")

            # 工具结果：Anthropic 放在 user 消息的 block 里
            if isinstance(content, list):
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if tool_results:
                    for tr in tool_results:
                        out = tr.get("content")
                        if not isinstance(out, str):
                            out = _text_of(out) or str(out)
                        messages.append(
                            Message(role="tool", content=out, tool_call_id=tr.get("tool_use_id"))
                        )
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if texts:
                        messages.append(Message(role=role, content="".join(texts)))
                    continue

                # 模型发起的 tool_use block
                tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                if tool_uses:
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    messages.append(
                        Message(
                            role="assistant",
                            content="".join(texts),
                            tool_calls=[
                                ToolCall(
                                    id=tu.get("id") or f"toolu_{i}",
                                    name=tu.get("name", ""),
                                    arguments=tu.get("input") or {},
                                )
                                for i, tu in enumerate(tool_uses)
                            ],
                        )
                    )
                    continue

            messages.append(Message(role=role, content=_text_of(content)))

        tools = None
        if body.get("tools"):
            tools = [
                ToolSpec(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("input_schema") or {},
                )
                for t in body["tools"]
            ] or None

        return ChatRequest(
            model=body.get("model") or settings.default_model,
            messages=messages,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_tokens"),
            stop=body.get("stop_sequences"),
            stream=bool(body.get("stream")),
            tools=tools,
            augmentations=body.get("augmentations") or [],
            session_id=(body.get("metadata") or {}).get("user_id"),
        )

    def render_response(self, resp: ChatResponse, request_id: str | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if resp.message.content:
            content.append({"type": "text", "text": resp.message.content})
        for tc in resp.message.tool_calls or []:
            content.append(
                {"type": "tool_use", "id": tc.id or new_id("toolu_"), "name": tc.name, "input": tc.arguments}
            )
        if not content:
            content.append({"type": "text", "text": ""})

        return {
            "id": request_id or new_id("msg_"),
            "type": "message",
            "role": "assistant",
            "model": resp.model,
            "content": content,
            "stop_reason": STOP_REASON_MAP.get(resp.finish_reason, "end_turn"),
            "stop_sequence": None,
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }

    async def stream_frames(
        self,
        events: AsyncIterator[dict[str, Any]],
        model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        mid = request_id or new_id("msg_")

        def emit(etype: str, data: dict[str, Any]) -> str:
            return self.sse({"type": etype, **data}, event=etype)

        yield emit(
            "message_start",
            {
                "message": {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            },
        )
        block_index = -1
        open_block = False
        calls: list[dict[str, Any]] = []
        finish = "stop"
        usage = {"input_tokens": 0, "output_tokens": 0}

        async for evt in events:
            if evt["type"] == "delta":
                if not open_block:
                    block_index += 1
                    open_block = True
                    yield emit(
                        "content_block_start",
                        {"index": block_index, "content_block": {"type": "text", "text": ""}},
                    )
                yield emit(
                    "content_block_delta",
                    {"index": block_index, "delta": {"type": "text_delta", "text": evt["text"]}},
                )
            elif evt["type"] == "tool_calls":
                calls = evt["calls"]
            elif evt["type"] == "done":
                finish = evt["finish_reason"]
                u = evt["usage"]
                usage = {"input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"]}

        if open_block:
            yield emit("content_block_stop", {"index": block_index})

        # 工具调用作为独立的 tool_use block 跟在后面
        for i, c in enumerate(calls):
            idx = block_index + 1 + i
            yield emit(
                "content_block_start",
                {
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": c["id"] or new_id("toolu_"),
                        "name": c["name"],
                        "input": {},
                    },
                },
            )
            import json

            yield emit(
                "content_block_delta",
                {
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(c["arguments"], ensure_ascii=False)},
                },
            )
            yield emit("content_block_stop", {"index": idx})

        yield emit(
            "message_delta",
            {
                "delta": {"stop_reason": STOP_REASON_MAP.get(finish, "end_turn"), "stop_sequence": None},
                "usage": usage,
            },
        )
        yield emit("message_stop", {})
