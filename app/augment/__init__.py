"""外挂层总装配。

一次对话的完整流水线：

    guard  ->  memory(读历史)  ->  rag(检索注入)  ->  tools(循环调用)  ->  memory(写历史)

拆成 ``prepare`` / ``finalize`` 两段，是为了让**流式**和**非流式**共用同一套前置
处理：流式只多了一个"边生成边吐"的过程，前后两段完全一致。

全部是**声明式**的：请求里带 ``augmentations: ["rag","tools","memory","guard"]``
就启用对应外挂，带空数组就是裸模型。
"""
from __future__ import annotations

import json
from typing import Any

from ..config import settings
from ..ollama_client import OllamaClient
from ..ollama_client import client as default_client
from ..schemas import ChatRequest, ChatResponse, Message
from . import memory, rag, tools
from .guard import GuardRejected, apply_guard  # noqa: F401  (对外暴露)

AVAILABLE = ("guard", "memory", "rag", "tools")
DEFAULT = ("guard", "rag", "tools")


def last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


def _inject_system(req: ChatRequest, extra: str) -> None:
    """把一段上下文追加到 system 消息里（没有就新建）。"""
    for m in req.messages:
        if m.role == "system":
            m.content = f"{m.content}\n\n{extra}" if m.content else extra
            return
    req.messages.insert(0, Message(role="system", content=extra, source="inject"))


def prepare(req: ChatRequest) -> dict[str, Any]:
    """执行所有"进模型之前"的外挂，就地修改 ``req``。"""
    enabled = [a for a in (req.augmentations or []) if a in AVAILABLE]
    diag: dict[str, Any] = {"enabled": enabled, "notes": []}

    # ---- 1. guard
    if "guard" in enabled:
        report = apply_guard(req)
        diag["guard"] = report.as_dict()
        if report.total_redacted:
            diag["notes"].append(f"已脱敏 {report.total_redacted} 处 PII")

    # ---- 2. memory：把历史拼回上下文
    if "memory" in enabled and req.session_id:
        history = memory.recent_messages(req.session_id)
        if history:
            hist_msgs = [Message(role=h["role"], content=h["content"], source="memory") for h in history]
            req.messages = hist_msgs + req.messages
            diag["memory"] = {"injected": len(hist_msgs)}
        else:
            diag["memory"] = {"injected": 0}

    # ---- 3. rag：拿最后一句用户输入去检索
    if "rag" in enabled:
        query = last_user_text(req.messages)
        context, hits = rag.build_rag_context(query) if query else ("", [])
        if context:
            _inject_system(req, context)
            diag["rag"] = {"hits": [{"source": h["source"], "score": round(h["score"], 3)} for h in hits]}
        else:
            diag["rag"] = {"hits": [], "note": "知识库无命中或为空"}

    # ---- 4. tools：补上工具定义
    if "tools" in enabled and settings.enable_tools and not req.tools:
        req.tools = tools.tool_specs()

    return diag


def finalize(req: ChatRequest, resp: ChatResponse, diag: dict[str, Any]) -> None:
    """执行"出模型之后"的外挂。"""
    enabled = diag.get("enabled") or []
    diag["tool_trace"] = resp.tool_trace
    if resp.tool_trace:
        diag["tool_rounds"] = len(resp.tool_trace)

    # memory：写回本轮
    if "memory" in enabled and req.session_id:
        memory.append_turn(req.session_id, last_user_text(req.messages), resp.message.content)

    effective = list(enabled)
    if resp.tool_trace:
        effective.append("tools")
    resp.augmentations_used = list(dict.fromkeys(effective))


async def run(
    req: ChatRequest,
    client: OllamaClient | None = None,
) -> tuple[ChatResponse, dict[str, Any]]:
    """非流式完整流水线。"""
    cli = client or default_client
    diag = prepare(req)

    resp = await cli.chat(req)
    if req.tools and resp.message.tool_calls:
        resp = await tools.run_tool_loop(req, resp, cli.chat)

    finalize(req, resp, diag)
    return resp, diag


async def run_stream(
    req: ChatRequest,
    client: OllamaClient | None = None,
):
    """流式流水线，产出内部事件流。

    工具循环在流式下同样成立：某一轮模型吐出 ``tool_calls`` 就执行工具、
    追加进 messages、再开一轮流式，直到模型给出最终文本。
    """
    cli = client or default_client
    diag = prepare(req)

    text_parts: list[str] = []
    total_prompt = total_completion = 0
    trace: list[dict[str, Any]] = []
    rounds = 0

    while True:
        calls: list[dict[str, Any]] = []
        done_evt: dict[str, Any] | None = None

        async for evt in cli.chat_stream(req):
            if evt["type"] == "delta":
                text_parts.append(evt["text"])
                yield evt
            elif evt["type"] == "tool_calls":
                calls = evt["calls"]
            elif evt["type"] == "done":
                done_evt = evt
                total_prompt += evt["usage"]["prompt_tokens"]
                total_completion += evt["usage"]["completion_tokens"]

        if not calls or not req.tools or rounds >= settings.max_tool_rounds:
            # 收尾
            finish = (done_evt or {}).get("finish_reason", "stop")
            yield {
                "type": "done",
                "finish_reason": "tool_calls" if calls else finish,
                "usage": {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_prompt + total_completion,
                },
                "augmentations_used": diag.get("enabled", []),
                "tool_trace": trace,
            }
            # 落库 memory
            from ..schemas import ChatResponse as _CR

            fake = _CR(
                model=req.model,
                message=Message(role="assistant", content="".join(text_parts)),
                finish_reason="stop",
                tool_trace=trace,
            )
            finalize(req, fake, diag)
            return

        # 执行工具，进入下一轮
        rounds += 1
        req.messages.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    tools.ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"]) for c in calls
                ],
            )
        )
        for c in calls:
            call = tools.ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            result = tools.execute_tool(call)
            trace.append({"round": rounds, "tool": call.name, "arguments": call.arguments, "result": result})
            req.messages.append(
                Message(
                    role="tool",
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call.id,
                    name=call.name,
                    source="tool",
                )
            )
