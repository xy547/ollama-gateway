"""全局配置。

所有配置项均可通过环境变量覆盖，前缀 ``OG_``。
例如 ``OG_OLLAMA_BASE_URL=http://192.168.1.10:11434``。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_prefix="OG_",
        extra="ignore",
    )

    # ---------- Ollama 上游 ----------
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen2.5:1.5b"
    request_timeout: float = 180.0

    # ---------- 对外服务 ----------
    # 逗号分隔的 API Key 列表；为空表示不鉴权（仅建议本机开发时使用）
    api_keys: str = ""
    # 每 Key 每分钟请求上限；<=0 表示不限
    rate_limit_per_minute: int = 60
    # 单次请求输入字符上限，超出直接 413
    max_input_chars: int = 20000
    service_name: str = "ollama-gateway"

    # ---------- 外挂 ----------
    enable_tools: bool = True
    max_tool_rounds: int = 4
    data_dir: str = str(BASE_DIR / "data")

    @property
    def key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def auth_enabled(self) -> bool:
        return bool(self.key_list)

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# 方便在脚本里直接改
if os.getenv("OG_DEBUG"):
    print(f"[config] ollama={settings.ollama_base_url} auth={'on' if settings.auth_enabled else 'off'}")
