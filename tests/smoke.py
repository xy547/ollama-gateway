"""冒烟测试：打真实运行中的服务，验证鉴权 / 限流 / 四协议 / 对比接口。

用法：
    python tests/smoke.py --base http://127.0.0.1:8000
    python tests/smoke.py --base http://127.0.0.1:8001 --key secret123

不依赖 pytest，直接跑，输出带颜色的通过/失败摘要，方便给别人演示。
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

PASS, FAIL = "\033[32m PASS \033[0m", "\033[31m FAIL \033[0m"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  -> {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="", help="API Key；给了就测试鉴权")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--skip-llm", action="store_true", help="只测不消耗模型的部分")
    ap.add_argument("--test-ratelimit", action="store_true", help="连打推理端点，确认限流会触发")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    hdrs = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    c = httpx.Client(base_url=base, timeout=180, headers=hdrs)

    # ---------- 运维 ----------
    r = c.get("/healthz")
    check(r.status_code == 200, "GET /healthz", str(r.status_code))

    r = c.get("/readyz")
    ready = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    check(r.status_code == 200, "GET /readyz", f"models={ready.get('models')}")

    # ---------- 鉴权 ----------
    if args.key:
        r = httpx.get(f"{base}/api/models", timeout=10)
        check(r.status_code == 401, "无 Key 被拒绝", str(r.status_code))
        r = httpx.get(f"{base}/api/models", headers={"Authorization": "Bearer wrong"}, timeout=10)
        check(r.status_code == 403, "错误 Key 被拒绝", str(r.status_code))
        r = httpx.get(f"{base}/api/models", headers={"x-api-key": args.key}, timeout=10)
        check(r.status_code == 200, "x-api-key 头可用", str(r.status_code))

    # ---------- 元信息 ----------
    r = c.get("/v1/meta/protocols")
    protos = [p["name"] for p in r.json().get("protocols", [])] if r.status_code == 200 else []
    check(len(protos) == 4, "四协议元信息", str(protos))

    r = c.get("/v1/meta/augmentations")
    augs = r.json().get("available", []) if r.status_code == 200 else []
    check(set(augs) == {"guard", "memory", "rag", "tools"}, "四外挂元信息", str(augs))

    r = c.get("/api/models")
    check(r.status_code == 200 and r.json().get("count", 0) > 0, "GET /api/models", f"{r.json().get('count')} 个")

    # ---------- 限流（故意打不存在的模型，秒回 400，不占 GPU）----------
    if args.test_ratelimit:
        codes = []
        for _ in range(8):
            rr = c.post("/v1/chat/completions",
                        json={"model": "no-such-model:1b", "messages": [{"role": "user", "content": "hi"}]})
            codes.append(rr.status_code)
        check(429 in codes, "限流触发 429", str(codes))

    if args.skip_llm:
        return summary()

    # ---------- 四协议各打一次 ----------
    prompt = "回答两个字：收到"
    cases = {
        "ollama": ("/api/chat", {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                                 "stream": False, "options": {"num_predict": 24}}),
        "openai-chat": ("/v1/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                                                 "max_tokens": 24, "stream": False}),
        "openai-responses": ("/v1/responses", {"model": args.model, "input": prompt,
                                               "max_output_tokens": 24, "stream": False}),
        "anthropic": ("/v1/messages", {"model": args.model, "system": "简洁", "max_tokens": 24,
                                        "messages": [{"role": "user", "content": prompt}], "stream": False}),
    }
    for name, (path, body) in cases.items():
        try:
            r = c.post(path, json=body)
            ok = r.status_code == 200 and _text(name, r.json())
            check(bool(ok), f"POST {path}", f"{r.status_code} / {len(_text(name, r.json()))} 字")
        except Exception as exc:  # noqa: BLE001
            check(False, f"POST {path}", str(exc)[:80])

    # ---------- 协议校验：Anthropic 缺 max_tokens 应 400 ----------
    r = c.post("/v1/messages", json={"model": args.model, "messages": [{"role": "user", "content": "hi"}]})
    check(r.status_code == 400, "Anthropic 缺 max_tokens 被拒", str(r.status_code))

    # ---------- 未知模型应 400 ----------
    r = c.post("/v1/chat/completions", json={"model": "no-such-model:1b",
                                             "messages": [{"role": "user", "content": "hi"}]})
    check(r.status_code == 400, "未知模型提示清晰", (r.json().get("error") or {}).get("message", "")[:50])

    # ---------- 外挂 ----------
    r = c.post("/v1/chat/completions", json={
        "model": args.model, "max_tokens": 120, "augmentations": ["tools"],
        "messages": [{"role": "user", "content": "现在几点？"}],
    })
    body = r.json()
    check(r.status_code == 200 and bool(body["choices"][0]["message"]["content"]), "tools 外挂",
          body["choices"][0]["message"]["content"][:30] if r.status_code == 200 else "")

    # ---------- 对比接口 ----------
    r = c.post("/api/compare", json={"prompt": "1+1 等于几？只回答数字", "model": args.model,
                                     "max_tokens": 24, "augmentations": []})
    if r.status_code == 200:
        d = r.json()
        oks = sum(1 for x in d["results"] if x.get("ok"))
        check(oks == 4, "POST /api/compare 四协议齐发", f"{oks}/4 成功")
        check(len(d.get("cheatsheet", [])) >= 10, "字段对照表", f"{len(d['cheatsheet'])} 行")
    else:
        check(False, "POST /api/compare", str(r.status_code))

    return summary()


def _text(protocol: str, d: dict) -> str:
    try:
        if protocol == "ollama":
            return d["message"]["content"]
        if protocol == "openai-chat":
            return d["choices"][0]["message"]["content"] or ""
        if protocol == "openai-responses":
            return "".join(b.get("text", "") for i in d["output"] if i["type"] == "message" for b in i["content"])
        if protocol == "anthropic":
            return "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
    except (KeyError, IndexError, TypeError):
        return ""
    return ""


def summary() -> int:
    ok = sum(1 for r in results if r[0])
    total = len(results)
    print("\n" + "=" * 56)
    print(f"结果：{ok}/{total} 通过" + ("  全部通过" if ok == total else "  有失败项"))
    for good, name, detail in results:
        if not good:
            print(f"  x {name}  {detail}")
    print("=" * 56)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
