"""端到端自检：不经过 HTTP，直接调内核，打印外挂是否真的生效。"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import augment
from app.schemas import ChatRequest, Message


async def main() -> None:
    # ---------- 1. tools ----------
    req = ChatRequest(
        model="qwen2.5:1.5b",
        messages=[Message(role="user", content="现在几点？今天星期几？")],
        max_tokens=200,
        augmentations=["tools"],
    )
    resp, diag = await augment.run(req)
    print("=" * 60)
    print("[tools] 回答：", resp.message.content)
    print("[tools] trace：", json.dumps(resp.tool_trace, ensure_ascii=False, indent=2))
    print("[tools] 轮数：", diag.get("tool_rounds"))
    print("[tools] usage：", resp.usage.model_dump())

    # ---------- 2. rag ----------
    req2 = ChatRequest(
        model="qwen2.5:1.5b",
        messages=[Message(role="user", content="BM25 里的 k1 和 b 分别控制什么？简短回答。")],
        max_tokens=180,
        augmentations=["rag"],
    )
    resp2, diag2 = await augment.run(req2)
    print("=" * 60)
    print("[rag] 命中：", json.dumps(diag2.get("rag"), ensure_ascii=False))
    print("[rag] 回答：", resp2.message.content)

    # ---------- 3. memory ----------
    sid = "selftest-memory"
    from app.augment import memory

    memory.clear_session(sid)
    r1 = ChatRequest(
        model="qwen2.5:1.5b",
        messages=[Message(role="user", content="我叫徐悦，正在做一个单词雨的小游戏。记住了吗？回答『记住了』即可。")],
        max_tokens=60,
        augmentations=["memory"],
        session_id=sid,
    )
    await augment.run(r1)
    r2 = ChatRequest(
        model="qwen2.5:1.5b",
        messages=[Message(role="user", content="我叫什么名字？在做哪个项目？")],
        max_tokens=100,
        augmentations=["memory"],
        session_id=sid,
    )
    resp3, diag3 = await augment.run(r2)
    print("=" * 60)
    print("[memory] 注入条数：", diag3.get("memory"))
    print("[memory] 第二轮回答：", resp3.message.content)

    # ---------- 4. guard ----------
    from app.augment.guard import redact

    text, counts = redact("我的手机 13812345678，邮箱 a.b@example.com，身份证 110101199003074512")
    print("=" * 60)
    print("[guard] 脱敏后：", text)
    print("[guard] 命中：", counts)


if __name__ == "__main__":
    asyncio.run(main())
