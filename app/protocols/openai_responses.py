"""OpenAI Responses API：``POST /v1/responses``。

OpenAI 2025 年主推的新接口，定位是"取代 Chat Completions"。它跟 Chat Completions
**不是同一个东西**，最容易踩的几个坑：

===========  ==========================  ============================
维度          Chat Completions            Responses
===========  ==========================  ============================
输入字段      ``messages``                ``input``（字符串或数组）
系统提示      ``messages[role=system]``   ``instructions``（顶层独立字段）
输出长度      ``max_tokens``              ``max_output_tokens``
输出结构      ``choices[0].message``      ``output[]`` 数组，再嵌 ``content[]``
文本位置      ``message.content``(str)    ``output[0].content[0].text``
结束标志      ``finish_reason``           ``status`` + ``incomplete_details``
工具定义      ``tools[].function.{...}``  ``tools[].{name,parameters}``（扁平）
工具结果      ``role=tool`` 消息          ``{"type":"function_call_output"}``
流式事件      无类型，统一 chunk         **有类型**的事件（``event:`` 行）
===========  ==========================  ============================
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..config import settings
from ..schemas import ChatRequest, ChatResponse, Message, ToolCall, ToolSpec
from .base import ProtocolAdapter, new_id, now_ts


class OpenAIResponsesAdapter(ProtocolAdapter):
    name = "openai-responses"
    path = "/v1/responses"
    auth_hint = "Authorization: Bearer <key>"

    # -------------------------------------------------------------- 入
    def parse_request(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> ChatRequest:
        raw_input = body.get("input")
        if raw_input is None or raw_input == "":
            raise ValueError("input 不能为空")

        messages: list[Message] = []
        # instructions 独立于 input，等价于 system
        if body.get("instructions"):
            messages.append(Message(role="system", content=body["instructions"]))

        if isinstance(raw_input, str):
            messages.append(Message(role="user", content=raw_input))
        elif isinstance(raw_input, list):
            for item in raw_input:
                if not isinstance(item, dict):
                    messages.append(Message(role="user", content=str(item)))
                    continue
                itype = item.get("type")
                if itype == "function_call_output":
                    # 工具执行结果回流
                    out = item.get("output")
                    if not isinstance(out, str):
                        out = json.dumps(out, ensure_ascii=False)
                    messages.append(
                        Message(role="tool", content=out, tool_call_id=item.get("call_id"))
                    )
                    continue
                if itype == "function_call":
                    messages.append(
                        Message(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id=item.get("call_id") or item.get("id") or "",
                                    name=item.get("name", ""),
                                    arguments=_maybe_json(item.get("arguments")),
                                )
                            ],
                        )
                    )
                    continue
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    buf = []
                    for blk in content:
                        if isinstance(blk, dict):
                            if blk.get("type") in ("input_text", "output_text", "text"):
                                buf.append(blk.get("text", ""))
                            elif blk.get("type") == "refusal":
                                buf.append(blk.get("refusal", ""))
                        elif isinstance(blk, str):
                            buf.append(blk)
                    content = "".join(buf)
                messages.append(Message(role=role, content=content or ""))
        else:
            raise ValueError("input 必须是字符串或数组")

        if not messages:
            raise ValueError("input 解析后没有任何消息")

        # tools 是扁平结构，与 Chat Completions 不同
        tools = None
        if body.get("tools"):
            tools = [
                ToolSpec(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("parameters") or {},
                )
                for t in body["tools"]
                if t.get("type", "function") == "function"
            ] or None

        # text.format.json_schema 是 Responses 的结构化输出写法
        schema = None
        fmt = ((body.get("text") or {}).get("format") or {})
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            schema = fmt.get("schema")

        return ChatRequest(
            model=body.get("model") or settings.default_model,
            messages=messages,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_output_tokens"),
            stream=bool(body.get("stream")),
            tools=tools,
            json_schema=schema,
            augmentations=body.get("augmentations") or [],
            session_id=body.get("user") or body.get("safety_identifier"),
        )

    # -------------------------------------------------------------- 出
    def render_response(self, resp: ChatResponse, request_id: str | None = None) -> dict[str, Any]:
        rid = request_id or new_id("resp_")
        output: list[dict[str, Any]] = []

        if resp.message.tool_calls:
            for tc in resp.message.tool_calls:
                output.append(
                    {
                        "type": "function_call",
                        "id": new_id("fc_"),
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        "status": "completed",
                    }
                )
        if resp.message.content or not output:
            output.append(
                {
                    "type": "message",
                    "id": new_id("msg_"),
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": resp.message.content,
                            "annotations": [],
                        }
                    ],
                }
            )

        return {
            "id": rid,
            "object": "response",
            "created_at": now_ts(),
            "status": "incomplete" if resp.finish_reason == "length" else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if resp.finish_reason == "length" else None,
            "model": resp.model,
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": resp.usage.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": resp.usage.total_tokens,
            },
            "error": None,
            "metadata": {},
        }

    async def stream_frames(
        self,
        events: AsyncIterator[dict[str, Any]],
        model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        rid = request_id or new_id("resp_")
        item_id = new_id("msg_")

        def emit(etype: str, data: dict[str, Any]) -> str:
            payload = {"type": etype, "sequence_number": 0, **data}
            return self.sse(payload, event=etype)

        yield emit(
            "response.created",
            {"response": {"id": rid, "object": "response", "status": "in_progress", "model": model}},
        )
        yield emit("response.in_progress", {"response": {"id": rid, "status": "in_progress"}})
        yield emit(
            "response.output_item.added",
            {"output_index": 0, "item": {"type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []}},
        )
        yield emit(
            "response.content_part.added",
            {"item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
        )

        text_buf: list[str] = []
        calls: list[dict[str, Any]] = []
        finish = "stop"
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        async for evt in events:
            if evt["type"] == "delta":
                text_buf.append(evt["text"])
                yield emit(
                    "response.output_text.delta",
                    {"item_id": item_id, "output_index": 0, "content_index": 0, "delta": evt["text"]},
                )
            elif evt["type"] == "tool_calls":
                calls = evt["calls"]
            elif evt["type"] == "done":
                finish = evt["finish_reason"]
                u = evt["usage"]
                usage = {
                    "input_tokens": u["prompt_tokens"],
                    "output_tokens": u["completion_tokens"],
                    "total_tokens": u["total_tokens"],
                }

        full = "".join(text_buf)
        yield emit(
            "response.output_text.done",
            {"item_id": item_id, "output_index": 0, "content_index": 0, "text": full},
        )
        yield emit(
            "response.content_part.done",
            {"item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": full, "annotations": []}},
        )
        yield emit(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": full, "annotations": []}],
                },
            },
        )
        yield emit(
            "response.completed",
            {
                "response": {
                    "id": rid,
                    "object": "response",
                    "status": "completed" if finish != "length" else "incomplete",
                    "model": model,
                    "output": [
                        {
                            "type": "message",
                            "id": item_id,
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": full, "annotations": []}],
                        }
                    ],
                    "usage": usage,
                }
            },
        )


def _maybe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except json.JSONDecodeError:
            return {"_raw": value}
    return {}
