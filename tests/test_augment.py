"""外挂层测试：分词、检索、护栏、工具安全。"""
from __future__ import annotations

import json

import pytest

from app.augment import guard, rag, tools
from app.schemas import ChatRequest, ChatResponse, Message, ToolCall, Usage


# ------------------------------------------------------------------ 分词
def test_中文按字和bigram切分():
    toks = rag.tokenize("梯度下降")
    assert "梯" in toks
    assert "梯度" in toks  # bigram
    assert "下降" in toks


def test_英文按词切分且小写化():
    toks = rag.tokenize("BM25 Ranking")
    assert "bm25" in toks
    assert "ranking" in toks


def test_中英混排都能切出来():
    toks = rag.tokenize("用 Ollama 跑 Qwen2.5 模型")
    assert "ollama" in toks
    assert "qwen2.5" in toks
    assert "模型" in toks


# ------------------------------------------------------------------ 切块
def test_切块不会丢内容():
    text = "\n\n".join([f"第{i}段内容，" * 10 for i in range(12)])
    chunks = rag.split_chunks(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_短文档只有一块():
    chunks = rag.split_chunks("这是一小段话。", size=400)
    assert chunks == ["这是一小段话。"]


# ------------------------------------------------------------------ 检索
@pytest.fixture
def store(tmp_path):
    (tmp_path / "a.md").write_text(
        "# 检索算法\n\nBM25 是经典的相关性排序函数，用词频和逆文档频率打分。k1 控制词频饱和，b 控制长度归一化。",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "# 分布式训练\n\n数据并行会把 batch 切分到多张卡上，梯度通过 all-reduce 同步。",
        encoding="utf-8",
    )
    s = rag.BM25Store(tmp_path)
    s.build(force=True)
    return s


def test_bm25能检索到相关文档(store):
    hits = store.search("BM25 的 k1 是什么", top_k=2)
    assert hits
    assert hits[0]["source"] == "a.md"
    assert "k1" in hits[0]["text"]


def test_bm25无关查询不返回结果(store):
    assert store.search("量子纠缠退相干", top_k=3) == []


def test_统计信息正确(store):
    st = store.stats()
    assert st["documents"] == 2
    assert st["chunks"] >= 2
    assert st["vocab"] > 0


def test_写入文档后立即可检索(store):
    store.add_document("c.md", "# 新文档\n\n这里讲的是向量数据库和 HNSW 索引。")
    hits = store.search("HNSW 索引", top_k=2)
    assert any(h["source"] == "c.md" for h in hits)


def test_文件名不能目录穿越(store):
    out = store.add_document("../../evil.md", "坏东西")
    assert out["saved"] == "evil.md"


# ------------------------------------------------------------------ 护栏
def test_手机号被脱敏():
    text, counts = guard.redact("联系我 13812345678")
    assert "13812345678" not in text
    assert counts["phone"] == 1


def test_身份证和邮箱被脱敏():
    text, counts = guard.redact("身份证 110101199003074512，邮箱 zhang@example.com")
    assert "110101199003074512" not in text
    assert "zhang@example.com" not in text
    assert counts["id_card"] == 1
    assert counts["email"] == 1


def test_普通数字不误伤():
    text, counts = guard.redact("订单号 20240506，金额 128 元")
    assert text == "订单号 20240506，金额 128 元"
    assert counts == {}


def test_超长输入被拒绝(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_input_chars", 10)
    req = ChatRequest(model="m", messages=[Message(role="user", content="x" * 100)])
    with pytest.raises(guard.GuardRejected):
        guard.apply_guard(req)


# ------------------------------------------------------------------ 计算器安全
def test_计算器算得对():
    assert tools.execute_tool(ToolCall(id="1", name="calculate", arguments={"expression": "(23*7+5)/2"}))["result"] == 83


def test_计算器拒绝危险表达式():
    out = tools.execute_tool(ToolCall(id="1", name="calculate", arguments={"expression": "__import__('os').system('dir')"}))
    assert "error" in out


def test_计算器拒绝变量赋值():
    out = tools.execute_tool(ToolCall(id="1", name="calculate", arguments={"expression": "a=1"}))
    assert "error" in out


# ------------------------------------------------------------------ 工具注册表
def test_注册表里每个工具都有描述和参数():
    for name, item in tools.TOOL_REGISTRY.items():
        assert item["description"], name
        assert isinstance(item["parameters"], dict), name
        assert item["parameters"].get("type") == "object", name


def test_未知工具返回可用清单而不是崩溃():
    out = tools.execute_tool(ToolCall(id="1", name="not_exist", arguments={}))
    assert "error" in out
    assert "calculate" in out["error"]


def test_参数不匹配时返回可读错误():
    out = tools.execute_tool(ToolCall(id="1", name="calculate", arguments={"wrong": 1}))
    assert "error" in out


def test_tool_specs可筛选():
    only = tools.tool_specs(["calculate"])
    assert len(only) == 1
    assert only[0].name == "calculate"


# ------------------------------------------------------------------ 工具循环
async def test_工具循环会执行并二次调用(monkeypatch):
    """第一次返回 tool_calls，第二次返回最终文本，确认循环真的跑了两轮。"""
    calls: list[ChatRequest] = []

    async def fake_model(req: ChatRequest) -> ChatResponse:
        calls.append(req)
        if len(calls) == 1:
            return ChatResponse(
                model=req.model,
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="c1", name="calculate", arguments={"expression": "2+2"})],
                ),
                finish_reason="tool_calls",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        return ChatResponse(
            model=req.model,
            message=Message(role="assistant", content="答案是 4"),
            usage=Usage(prompt_tokens=20, completion_tokens=6, total_tokens=26),
        )

    req = ChatRequest(model="m", messages=[Message(role="user", content="2+2 等于几")], tools=tools.tool_specs())
    first = await fake_model(req)
    resp = await tools.run_tool_loop(req, first, fake_model)

    assert resp.message.content == "答案是 4"
    assert len(resp.tool_trace) == 1
    assert resp.tool_trace[0]["tool"] == "calculate"
    assert resp.tool_trace[0]["result"]["result"] == 4
    # 两轮用量应累加
    assert resp.usage.prompt_tokens == 30
    assert resp.usage.total_tokens == 41
    # 工具结果作为 role=tool 追加进了消息
    assert any(m.role == "tool" and "4" in m.content for m in req.messages)


async def test_工具循环有轮数上限(monkeypatch):
    """模型无限要工具时不能死循环。"""
    from app.config import settings

    monkeypatch.setattr(settings, "max_tool_rounds", 2)
    count = 0

    async def always_tool(req: ChatRequest) -> ChatResponse:
        nonlocal count
        count += 1
        return ChatResponse(
            model=req.model,
            message=Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id=f"c{count}", name="get_current_time", arguments={})],
            ),
            finish_reason="tool_calls",
        )

    req = ChatRequest(model="m", messages=[Message(role="user", content="时间")], tools=tools.tool_specs())
    first = await always_tool(req)
    await tools.run_tool_loop(req, first, always_tool)
    # 首次 1 次 + 循环 2 次
    assert count == 3
