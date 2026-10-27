"""四个协议适配器的注册表。"""
from __future__ import annotations

from .anthropic_messages import AnthropicMessagesAdapter
from .base import ProtocolAdapter
from .ollama_native import OllamaNativeAdapter
from .openai_chat import OpenAIChatAdapter
from .openai_responses import OpenAIResponsesAdapter

ADAPTERS: dict[str, ProtocolAdapter] = {
    a.name: a
    for a in (
        OllamaNativeAdapter(),
        OpenAIChatAdapter(),
        OpenAIResponsesAdapter(),
        AnthropicMessagesAdapter(),
    )
}


def get_adapter(name: str) -> ProtocolAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"未知协议 {name}，可选：{', '.join(ADAPTERS)}")
    return ADAPTERS[name]


def all_adapters() -> list[ProtocolAdapter]:
    return list(ADAPTERS.values())


def protocol_catalog() -> list[dict[str, str]]:
    """给 /v1/meta/protocols 用的元信息。"""
    return [
        {"name": a.name, "path": a.path, "media_type": a.stream_media_type, "auth": a.auth_hint}
        for a in ADAPTERS.values()
    ]


__all__ = ["ADAPTERS", "get_adapter", "all_adapters", "protocol_catalog", "ProtocolAdapter"]
