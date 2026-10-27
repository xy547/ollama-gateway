"""命令行版四协议对比。

先启动服务，然后：

    python scripts/compare_protocols.py "用一句话解释什么是 BM25"
    python scripts/compare_protocols.py --protocols anthropic,ollama "你好"
    python scripts/compare_protocols.py --show-raw "1+1"

它会把你这句话"分别用四种协议的写法"发一遍，然后横向打印：
文本、耗时、token、结束原因，以及各家响应体里文本到底躺在哪一层。
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

W = 46


def rule(ch: str = "-") -> None:
    print(ch * (W * 2 + 6))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", help="要问的话")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--key", default="", help="若服务开启了鉴权，填 API Key")
    ap.add_argument("--protocols", default="ollama,openai-chat,openai-responses,anthropic")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--show-raw", action="store_true", help="打印完整响应 JSON")
    args = ap.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else "用两句话解释什么是 API"
    body = {
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "augmentations": [],
        "protocols": [p.strip() for p in args.protocols.split(",") if p.strip()],
    }
    if args.model:
        body["model"] = args.model

    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    print(f"\n查询：{prompt}\n")

    try:
        r = httpx.post(f"{args.base}/api/compare", json=body, headers=headers, timeout=600)
    except httpx.ConnectError:
        print(f"连不上 {args.base}，先运行 python run.py", file=sys.stderr)
        return 1
    if r.status_code != 200:
        print(f"请求失败 {r.status_code}: {r.text[:400]}", file=sys.stderr)
        return 1

    data = r.json()
    print(f"模型：{data['model']}")
    rule("=")

    for res in data["results"]:
        if not res.get("ok"):
            print(f"\n[{res['protocol']}] 失败：{res.get('error', '')[:200]}")
            continue
        u = res["usage"]
        print(f"\n{res['protocol']}   POST {res['path']}")
        print(f"  文本    : {(res['text'] or '（空）').strip()[:300]}")
        print(f"  耗时    : {res['latency_ms']} ms")
        print(f"  token   : 输入 {u.get('prompt', 0)} / 输出 {u.get('completion', 0)}")
        print(f"  结束原因: {res['finish']}")
        print(f"  文本位置: {_text_path(res['protocol'])}")
        if args.show_raw:
            print("  原始响应:")
            print("   " + json.dumps(res["response"], ensure_ascii=False, indent=2).replace("\n", "\n   "))

    # ---------- 字段对照表 ----------
    print()
    rule("=")
    print("\n字段对照（同一件事，四种说法）\n")
    cols = ["Ollama", "OpenAI Chat", "Responses", "Anthropic"]
    header = f"{'维度':<10}" + "".join(f"{c:<30}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in data["cheatsheet"]:
        name, vals = row[0], row[1]
        print(f"{name:<10}" + "".join(f"{_ellip(v, 28):<30}" for v in vals))
    print()
    return 0


def _text_path(protocol: str) -> str:
    return {
        "ollama": "response.message.content",
        "openai-chat": "response.choices[0].message.content",
        "openai-responses": "response.output[0].content[0].text",
        "anthropic": "response.content[0].text",
    }.get(protocol, "?")


def _ellip(s: str, n: int) -> str:
    s = str(s).replace("\n", " / ")
    return s if len(s) <= n else s[: n - 1] + "…"


if __name__ == "__main__":
    sys.exit(main())
