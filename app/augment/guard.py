"""外挂四：输入护栏。

对外提供服务时，这一步是"把模型关进笼子"的第一道闸：
- 超长输入直接拒绝（防止有人用 10 万字把显存打爆）；
- 手机号 / 身份证 / 邮箱 / 银行卡等 PII 在**发给上游之前**脱敏，
  避免用户隐私被写进 Ollama 的日志或缓存在远端。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import settings
from ..schemas import ChatRequest, Message

PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
}


class GuardRejected(ValueError):
    """输入被护栏拦截。"""

    def __init__(self, message: str, status_code: int = 413):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class GuardReport:
    chars: int = 0
    redacted: dict[str, int] = field(default_factory=dict)

    @property
    def total_redacted(self) -> int:
        return sum(self.redacted.values())

    def as_dict(self) -> dict:
        return {"chars": self.chars, "redacted": self.redacted, "total_redacted": self.total_redacted}


def redact(text: str) -> tuple[str, dict[str, int]]:
    """把 PII 替换成占位符，返回 (新文本, 各类型命中次数)。"""
    counts: dict[str, int] = {}
    for name, pattern in PATTERNS.items():
        def _sub(m: re.Match[str], _n: str = name) -> str:
            counts[_n] = counts.get(_n, 0) + 1
            return f"[REDACTED:{_n}]"

        text = pattern.sub(_sub, text)
    return text, counts


def apply_guard(req: ChatRequest) -> GuardReport:
    """就地清洗 ``req``，超长则抛 ``GuardRejected``。"""
    total = sum(len(m.content or "") for m in req.messages)
    if total > settings.max_input_chars:
        raise GuardRejected(
            f"输入长度 {total} 字符，超过上限 {settings.max_input_chars}。"
            "请缩短上下文，或调大 OG_MAX_INPUT_CHARS。"
        )

    report = GuardReport(chars=total)
    new_messages: list[Message] = []
    for m in req.messages:
        if m.role in ("user", "system") and m.content:
            cleaned, counts = redact(m.content)
            for k, v in counts.items():
                report.redacted[k] = report.redacted.get(k, 0) + v
            new_messages.append(m.model_copy(update={"content": cleaned}))
        else:
            new_messages.append(m)
    req.messages = new_messages
    return report
