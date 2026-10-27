"""一键启动。

    python run.py                 # 默认 0.0.0.0:8000
    python run.py --port 9000     # 换端口
    python run.py --reload        # 开发模式，改代码自动重启
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Ollama Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（对外服务用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    import uvicorn

    def display_width(s: str) -> int:
        """中文字符在终端占两列，按显示宽度算才能对齐。"""
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)

    def row(label: str, value: str) -> str:
        pad = max(0, 52 - display_width(label) - display_width(value))
        return f"│  {label}{' ' * pad}{value}│"

    print(
        "\n"
        "┌" + "─" * 56 + "┐\n"
        + row("Ollama", "Gateway —— 四协议统一的本地模型服务")
        + "\n"
        + row("文档", f"http://127.0.0.1:{args.port}/docs")
        + "\n"
        + row("演示台", f"http://127.0.0.1:{args.port}/ui")
        + "\n"
        + row("四协议", "/api/chat  /v1/chat/completions")
        + "\n"
        + row("", "/v1/responses  /v1/messages")
        + "\n"
        + "└" + "─" * 56 + "┘\n"
    )
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
