"""外挂一：工具调用（function calling）。

模型本身只会生成文字，它能"查天气""算数""读文档"，靠的是我们把工具列表塞进
请求、模型输出一个 ``tool_calls``、我们再真的去执行、把结果塞回去让它继续。

这里把「工具注册」和「执行循环」拆开：
- ``TOOL_REGISTRY``：有哪些工具、怎么执行（同步函数即可）；
- ``run_tool_loop``：多轮循环，最多 ``max_tool_rounds`` 轮，防止模型自嗨。
"""
from __future__ import annotations

import ast
import json
import operator
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

from ..config import settings
from ..schemas import ChatRequest, ChatResponse, Message, ToolCall, ToolSpec

# ---------------------------------------------------------------- 安全计算器
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("只支持数字")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


# ---------------------------------------------------------------- 工具实现
def tool_get_current_time(tz_offset_hours: float = 8) -> dict[str, Any]:
    """返回指定时区的当前时间（默认东八区）。"""
    tz = timezone(timedelta(hours=tz_offset_hours))
    now = datetime.now(tz)
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
        "timezone": f"UTC{'+' if tz_offset_hours >= 0 else ''}{tz_offset_hours:g}",
    }


def tool_calculate(expression: str) -> dict[str, Any]:
    """计算一个算术表达式。"""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree)
        if value.is_integer():
            value = int(value)  # type: ignore[assignment]
        return {"expression": expression, "result": value}
    except Exception as exc:  # noqa: BLE001
        return {"expression": expression, "error": f"无法计算：{exc}"}


def tool_search_documents(query: str, top_k: int = 3) -> dict[str, Any]:
    """在本地知识库里检索（RAG 的检索部分，可被 tools 外挂复用）。"""
    from .rag import get_store

    hits = get_store().search(query, top_k=top_k)
    if not hits:
        return {"query": query, "hits": [], "note": "知识库为空，请先在 data/docs 下放 .md/.txt 文件"}
    return {
        "query": query,
        "hits": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]} for h in hits],
    }


def tool_list_models() -> dict[str, Any]:
    """列出本机已安装的 Ollama 模型。"""
    import httpx

    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = r.json().get("models", [])
        return {"count": len(models), "models": [m["name"] for m in models]}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"查询失败：{exc}"}


def tool_read_note(filename: str) -> dict[str, Any]:
    """读取知识库目录下的某个文件全文。"""
    base = Path(settings.data_path) / "docs"
    target = (base / filename).resolve()
    if not str(target).startswith(str(base.resolve())):
        return {"error": "路径越界，拒绝访问"}
    if not target.exists():
        return {"error": f"文件不存在：{filename}"}
    return {"filename": filename, "content": target.read_text(encoding="utf-8", errors="ignore")[:4000]}


def tool_get_weather(city: str) -> dict[str, Any]:
    """查询城市天气（演示用，返回模拟数据，不联网）。"""
    # 明确标注 mock，避免模型把它当真实数据讲给用户
    seed = sum(ord(c) for c in city) % 100
    return {
        "city": city,
        "temperature_c": 12 + seed % 18,
        "condition": ["晴", "多云", "小雨", "阴"][seed % 4],
        "source": "mock（演示数据，未接入真实气象 API）",
    }


# ---------------------------------------------------------------- 注册表
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "get_current_time": {
        "fn": tool_get_current_time,
        "description": "获取当前日期、时间与星期。用户问到'今天几号''现在几点''星期几'时必须调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "tz_offset_hours": {
                    "type": "number",
                    "description": "时区偏移小时数，中国为 8（默认）",
                }
            },
        },
    },
    "calculate": {
        "fn": tool_calculate,
        "description": "计算算术表达式。涉及数字运算时使用，不要自己心算。",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "如 (23*7+5)/2"}},
            "required": ["expression"],
        },
    },
    "search_documents": {
        "fn": tool_search_documents,
        "description": "在本地知识库中检索资料。回答需要引用私有文档时先调用它。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题"},
                "top_k": {"type": "integer", "description": "返回条数，默认 3"},
            },
            "required": ["query"],
        },
    },
    "list_models": {
        "fn": tool_list_models,
        "description": "列出本机已安装的 Ollama 模型清单。",
        "parameters": {"type": "object", "properties": {}},
    },
    "read_note": {
        "fn": tool_read_note,
        "description": "读取知识库目录下指定文件的完整内容。",
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string", "description": "文件名，如 notes.md"}},
            "required": ["filename"],
        },
    },
    "get_weather": {
        "fn": tool_get_weather,
        "description": "查询城市天气（演示用模拟数据）。",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}


def tool_specs(names: list[str] | None = None) -> list[ToolSpec]:
    """把注册表转成给模型的工具定义。"""
    keys = names or list(TOOL_REGISTRY)
    specs = []
    for k in keys:
        item = TOOL_REGISTRY.get(k)
        if not item:
            continue
        specs.append(
            ToolSpec(
                name=k,
                description=item["description"],
                parameters=item["parameters"],
            )
        )
    return specs


def execute_tool(call: ToolCall) -> dict[str, Any]:
    """执行一次工具调用，永远返回 dict（出错也返回 dict，让模型能自我纠正）。"""
    item = TOOL_REGISTRY.get(call.name)
    if not item:
        return {"error": f"没有名为 {call.name} 的工具，可用工具：{', '.join(TOOL_REGISTRY)}"}
    fn: Callable[..., Any] = item["fn"]
    try:
        return fn(**call.arguments)
    except TypeError as exc:
        return {"error": f"参数不匹配：{exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"工具执行失败：{exc}"}


# ---------------------------------------------------------------- 执行循环
async def run_tool_loop(
    req: ChatRequest,
    first: ChatResponse,
    call_model,
) -> ChatResponse:
    """模型要调工具就执行，把结果喂回去，直到它给出最终回答。

    ``call_model`` 是协程函数，签名 ``(ChatRequest) -> ChatResponse``。
    这里不直接依赖 OllamaClient，是为了单测时可以注入假的模型。
    """
    trace: list[dict[str, Any]] = []
    resp = first
    total_usage = resp.usage
    rounds = 0

    while resp.message.tool_calls and rounds < settings.max_tool_rounds:
        rounds += 1
        req.messages.append(
            Message(role="assistant", content=resp.message.content, tool_calls=resp.message.tool_calls)
        )
        for call in resp.message.tool_calls:
            started = time.perf_counter()
            result = execute_tool(call)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            trace.append(
                {"round": rounds, "tool": call.name, "arguments": call.arguments, "result": result, "ms": elapsed}
            )
            req.messages.append(
                Message(
                    role="tool",
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call.id,
                    name=call.name,
                    source="tool",
                )
            )
        resp = await call_model(req)
        total_usage = total_usage.merge(resp.usage)

    resp.usage = total_usage
    resp.tool_trace = trace
    if trace:
        resp.augmentations_used = list(dict.fromkeys(resp.augmentations_used + ["tools"]))
    return resp
