"""内部统一的对话模型。

四种对外协议（Ollama / OpenAI Chat / OpenAI Responses / Anthropic）都先被翻译成
这里的 ``ChatRequest``，再由 ``OllamaClient`` 调上游，最后翻译回各自的响应格式。

放一层内部模型的价值：
- 协议适配器只关心「我的格式 <-> 内部格式」，互不影响；
- 外挂（tools / RAG / memory）只需要在内核这一层实现一次；
- 加新协议不用改内核。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """模型发起的一次工具调用。"""

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    # assistant 发起工具调用时有值
    tool_calls: list[ToolCall] | None = None
    # role == "tool" 时，指向它回应的那个 tool_call
    tool_call_id: str | None = None
    name: str | None = None
    # 供外挂注入用：这条消息来自哪个增强模块（rag / memory / tool）
    source: str | None = None


class ToolSpec(BaseModel):
    """工具定义（内部统一形态）。"""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ChatRequest(BaseModel):
    """内部统一请求。"""

    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    stream: bool = False
    tools: list[ToolSpec] | None = None
    # 结构化输出：直接给 JSON Schema
    json_schema: dict[str, Any] | None = None
    # 外挂开关，见 augment/registry.py
    augmentations: list[str] = Field(default_factory=list)
    # 会话 ID（memory 外挂用）
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """内部统一响应。"""

    model: str
    message: Message
    finish_reason: Literal["stop", "length", "tool_calls", "error"] = "stop"
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    # 本次实际启用 / 触发的外挂，回传给调用方便于观测
    augmentations_used: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    upstream_model: str | None = None
