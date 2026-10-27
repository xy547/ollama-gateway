"""协议适配器的纯逻辑测试 —— 不连模型，只验证"翻译"是否正确。

这层是最容易写错又最容易测的地方，所以覆盖得细一点：
字段名差异、必填校验、工具参数的字符串/对象转换、结构的嵌套层级。
"""
from __future__ import annotations

import json

import pytest

from app.protocols import ADAPTERS, get_adapter
from app.schemas import ChatResponse, Message, ToolCall, Usage


# ------------------------------------------------------------------ Ollama 原生
def test_ollama_options_映射到统一字段():
    a = get_adapter("ollama")
    req = a.parse_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.5, "num_predict": 128, "top_p": 0.9},
        }
    )
    assert req.temperature == 0.5
    assert req.max_tokens == 128  # num_predict -> max_tokens
    assert req.top_p == 0.9
    assert req.messages[0].content == "hi"


def test_ollama_空messages报错():
    a = get_adapter("ollama")
    with pytest.raises(ValueError, match="messages"):
        a.parse_request({"model": "m"})


# ------------------------------------------------------------------ OpenAI Chat
def test_openai_chat_工具参数是json字符串要能解析():
    a = get_adapter("openai-chat")
    req = a.parse_request(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "算一下"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": '{"expression":"1+1"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"result":2}'},
            ],
        }
    )
    assert req.messages[1].tool_calls[0].arguments == {"expression": "1+1"}
    assert req.messages[2].tool_call_id == "call_1"


def test_openai_chat_多段content能拼成纯文本():
    a = get_adapter("openai-chat")
    req = a.parse_request(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]}
            ],
        }
    )
    assert req.messages[0].content == "第一段第二段"


def test_openai_chat_响应结构符合规范():
    a = get_adapter("openai-chat")
    resp = ChatResponse(
        model="m",
        message=Message(role="assistant", content="你好"),
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    out = a.render_response(resp)
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "你好"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 5
    assert out["id"].startswith("chatcmpl-")


def test_openai_chat_工具调用的arguments是字符串():
    a = get_adapter("openai-chat")
    resp = ChatResponse(
        model="m",
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})],
        ),
        finish_reason="tool_calls",
    )
    out = a.render_response(resp)
    call = out["choices"][0]["message"]["tool_calls"][0]
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"a": 1}
    assert out["choices"][0]["finish_reason"] == "tool_calls"


# ------------------------------------------------------------------ Responses
def test_responses_input可以是字符串():
    a = get_adapter("openai-responses")
    req = a.parse_request({"model": "m", "input": "你好", "instructions": "你是助手"})
    assert req.messages[0].role == "system"
    assert req.messages[0].content == "你是助手"  # instructions 独立字段
    assert req.messages[1].role == "user"
    assert req.messages[1].content == "你好"


def test_responses_input可以是数组且带工具结果():
    a = get_adapter("openai-responses")
    req = a.parse_request(
        {
            "model": "m",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "查时间"}]},
                {"type": "function_call_output", "call_id": "call_9", "output": "2026-01-01"},
            ],
        }
    )
    assert req.messages[0].content == "查时间"
    assert req.messages[1].role == "tool"
    assert req.messages[1].tool_call_id == "call_9"


def test_responses_max_output_tokens映射正确():
    a = get_adapter("openai-responses")
    req = a.parse_request({"model": "m", "input": "hi", "max_output_tokens": 77})
    assert req.max_tokens == 77


def test_responses_响应文本在output嵌套里():
    a = get_adapter("openai-responses")
    resp = ChatResponse(model="m", message=Message(role="assistant", content="答案"))
    out = a.render_response(resp)
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"][0]["type"] == "output_text"
    assert out["output"][0]["content"][0]["text"] == "答案"
    assert out["usage"]["input_tokens"] == 0


def test_responses_长度截断是incomplete():
    a = get_adapter("openai-responses")
    resp = ChatResponse(model="m", message=Message(role="assistant", content="x"), finish_reason="length")
    out = a.render_response(resp)
    assert out["status"] == "incomplete"
    assert out["incomplete_details"]["reason"] == "max_output_tokens"


# ------------------------------------------------------------------ Anthropic
def test_anthropic_缺max_tokens必须报错():
    a = get_adapter("anthropic")
    with pytest.raises(ValueError, match="max_tokens"):
        a.parse_request({"model": "m", "messages": [{"role": "user", "content": "hi"}]})


def test_anthropic_system是顶层字段():
    a = get_adapter("anthropic")
    req = a.parse_request({"model": "m", "system": "你是助手", "max_tokens": 10,
                           "messages": [{"role": "user", "content": "hi"}]})
    assert req.messages[0].role == "system"
    assert req.messages[0].content == "你是助手"


def test_anthropic_工具定义用input_schema():
    a = get_adapter("anthropic")
    req = a.parse_request(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        }
    )
    assert req.tools[0].name == "f"
    assert req.tools[0].parameters == {"type": "object", "properties": {}}


def test_anthropic_tool_result在user消息里():
    a = get_adapter("anthropic")
    req = a.parse_request(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "几点了"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "12:00"}],
                },
            ],
        }
    )
    roles = [m.role for m in req.messages]
    assert "tool" in roles
    tool_msg = next(m for m in req.messages if m.role == "tool")
    assert tool_msg.content == "12:00"
    assert tool_msg.tool_call_id == "toolu_1"


def test_anthropic_stop_reason取值映射():
    a = get_adapter("anthropic")
    for fr, expect in [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use")]:
        out = a.render_response(ChatResponse(model="m", message=Message(role="assistant", content="x"), finish_reason=fr))
        assert out["stop_reason"] == expect


def test_anthropic_响应是block数组():
    a = get_adapter("anthropic")
    out = a.render_response(ChatResponse(model="m", message=Message(role="assistant", content="你好")))
    assert out["type"] == "message"
    assert out["content"][0] == {"type": "text", "text": "你好"}
    assert out["usage"]["input_tokens"] == 0


# ------------------------------------------------------------------ 全局
def test_四个协议路径互不冲突():
    paths = [a.path for a in ADAPTERS.values()]
    assert len(set(paths)) == 4
    assert set(paths) == {"/api/chat", "/v1/chat/completions", "/v1/responses", "/v1/messages"}
