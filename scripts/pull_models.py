"""模型下载脚本（不依赖 ollama CLI，直接打 HTTP API）。

为什么不用 `ollama pull`？因为 ``OLLAMA_HOST=0.0.0.0`` 这类环境下 CLI 会连不上，
而 HTTP API 一直是稳的——顺便这也是本项目"对外服务"思路的起点。

    python scripts/pull_models.py qwen2.5:1.5b
    python scripts/pull_models.py --list
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

BASE = "http://127.0.0.1:11434"


def list_models() -> None:
    r = httpx.get(f"{BASE}/api/tags", timeout=10)
    data = r.json()
    print(f"{'模型':<28}{'参数':<10}{'量化':<10}{'大小':<8}能力")
    print("-" * 70)
    for m in data.get("models", []):
        d = m.get("details") or {}
        size = (m.get("size") or 0) / 1e9
        caps = ",".join(m.get("capabilities") or [])
        print(f"{m['name']:<28}{d.get('parameter_size', ''):<10}{d.get('quantization_level', ''):<10}{size:<8.2f}{caps}")


def pull(model: str) -> None:
    print(f"拉取 {model} …")
    last_pct = -1
    with httpx.stream("POST", f"{BASE}/api/pull", json={"model": model, "stream": True}, timeout=3600) as r:
        for line in r.iter_lines():
            if not line.strip():
                continue
            evt = json.loads(line)
            if evt.get("error"):
                print(f"\n失败：{evt['error']}")
                return
            total, done = evt.get("total"), evt.get("completed")
            if total and done:
                pct = int(done / total * 100)
                if pct != last_pct and pct % 5 == 0:
                    last_pct = pct
                    bar = "#" * (pct // 4) + "." * (25 - pct // 4)
                    print(f"\r  [{bar}] {pct:>3}%  {done / 1e6:>8.1f}/{total / 1e6:.1f} MB", end="")
            elif evt.get("status"):
                print(f"\r  {evt['status']:<60}", end="")
    print("\n完成")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="要下载的模型名")
    ap.add_argument("--list", action="store_true", help="只列出已有模型")
    args = ap.parse_args()

    try:
        httpx.get(BASE, timeout=5)
    except Exception:
        print(f"连不上 Ollama（{BASE}），请先运行 ollama serve", file=sys.stderr)
        return 1

    if args.list or not args.models:
        list_models()
        return 0
    for m in args.models:
        pull(m)
    print()
    list_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())
