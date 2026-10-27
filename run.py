"""一键启动。

    python run.py                 # 默认 0.0.0.0:8000
    python run.py --port 9000     # 换端口
    python run.py --reload        # 开发模式，改代码自动重启
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Ollama Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（对外服务用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    import uvicorn

    print(
        f"""
┌──────────────────────────────────────────────────────────┐
│  Ollama Gateway                                          │
│  文档     http://127.0.0.1:{args.port}/docs{' ' * (28 - len(str(args.port)))}│
│  演示台   http://127.0.0.1:{args.port}/ui{' ' * (30 - len(str(args.port)))}│
│  四协议   /api/chat  /v1/chat/completions                │
│           /v1/responses  /v1/messages                    │
└──────────────────────────────────────────────────────────┘
"""
    )
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
