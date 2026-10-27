"""协议适配器基类。

一个适配器负责两件事：
1. **入**：把某种外部协议的请求体翻译成内部 ``ChatRequest``；
2. **出**：把内部 ``ChatResponse``（或流式事件流）渲染成该协议要求的响应格式。

加上这层之后，加新协议（比如 Gemini 的 generateContent）只需要写一个子类。
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ..schemas import ChatRequest, ChatResponse


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


def now_ts() -> int:
    return int(time.time())


class ProtocolAdapter(ABC):
    """协议适配器。"""

    name: str = ""
    #: 对外暴露的路径
    path: str = ""
    #: 流式响应使用的媒体类型
    stream_media_type: str = "text/event-stream"
    #: 该协议是否有实质的鉴权头要求（仅用于文档展示）
    auth_hint: str = ""

    # -------------------------------------------------------------- 入
    @abstractmethod
    def parse_request(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> ChatRequest:
        """外部请求体 -> 内部请求。参数非法时抛 ``ValueError``。"""

    # -------------------------------------------------------------- 出
    @abstractmethod
    def render_response(self, resp: ChatResponse, request_id: str | None = None) -> dict[str, Any]:
        """内部响应 -> 该协议的非流式响应体。"""

    @abstractmethod
    def stream_frames(
        self,
        events: AsyncIterator[dict[str, Any]],
        model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """把内部事件流渲染成该协议的流式帧（已经是可直接写出的字符串）。"""

    # -------------------------------------------------------------- 辅助
    def sse(self, data: dict[str, Any], event: str | None = None) -> str:
        """标准 SSE 帧。"""
        import json

        payload = json.dumps(data, ensure_ascii=False)
        if event:
            return f"event: {event}\ndata: {payload}\n\n"
        return f"data: {payload}\n\n"

    def ndjson(self, data: dict[str, Any]) -> str:
        """NDJSON 帧（Ollama 原生流式用的就是这个，不是 SSE）。"""
        import json

        return json.dumps(data, ensure_ascii=False) + "\n"
