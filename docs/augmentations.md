# 外挂实现详解

"外挂"（augmentation）指的是**在模型本体之外，由网关替模型完成的能力**：
模型本身只会吐字，是网关把工具、知识、记忆接到它嘴边。

四类外挂都是**声明式**的 —— 请求体里写 `augmentations: ["tools", "rag"]` 才启用，
不写就是裸模型。这样同一个服务既能当"严谨引用助手"，也能当"随便聊的玩具"。

---

## 流水线

```
                     ┌──────────────┐
   用户请求 ────────▶ │ prepare()    │  guard → memory(读) → rag → tools(定义)
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ Ollama 调用   │  /api/chat（可流式）
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ tools 循环    │  有 tool_calls 就执行并回喂，最多 N 轮
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ finalize()    │  memory(写) → 标注实际生效的外挂
                     └──────────────┘
```

`prepare` / `finalize` 拆成两段，是为了让**流式和非流式共用同一套前置处理**。

实现见 `app/augment/__init__.py`。

---

## 1. tools —— 工具调用

### 注册表

工具就是 `TOOL_REGISTRY` 里的一个条目（`app/augment/tools.py`）：

```python
"calculate": {
    "fn": tool_calculate,
    "description": "计算算术表达式。涉及数字运算时使用，不要自己心算。",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "如 (23*7+5)/2"}},
        "required": ["expression"],
    },
},
```

`description` 是写给模型看的，**质量直接决定调用率**。写"用于计算数学表达式"这种废话，
模型经常不调；写"涉及数字运算时使用，不要自己心算"，它就调。

### 内置工具

| 工具 | 作用 | 注意 |
|---|---|---|
| `get_current_time` | 当前日期/时间/星期 | 默认东八区 |
| `calculate` | 算术表达式求值 | **AST 白名单**，不 eval 字符串 |
| `search_documents` | 在知识库里检索 | RAG 的检索部分复用 |
| `read_note` | 读取知识库文件 | 做了路径越界检查 |
| `list_models` | 列出本机已装模型 | |
| `get_weather` | 城市天气 | **mock 数据**，响应里标了来源 |

### 安全

`calculate` 用 AST 解析 + 白名单节点（`ast.Add` / `ast.Mult` / `ast.Pow` …）求值，
不是 `eval()`。下面这些会被拒：

```python
__import__('os').system('dir')   # -> error: 不支持的表达式节点：Call
a = 1                            # -> error: 不支持的表达式节点：Assign
```

### 执行循环

```
第 1 轮：模型返回 tool_calls
  → 网关执行工具，把结果作为 role=tool 消息追加
第 2 轮：模型看到工具结果，给出最终回答
```

- 轮数上限 `max_tool_rounds`（默认 4）。模型如果一直要工具，到上限就停，不会死循环。
- 工具执行失败**不抛异常**，而是把 `{"error": "..."}` 回给模型，让它自己纠正 —— 小模型经常传错参数名。
- 每次调用都记进 `tool_trace`，调试时能看清"它到底调了什么、参数对不对、耗时多少"。

流式模式下同样支持工具循环：某一轮吐了 `tool_calls` 就执行、开新一轮继续流。

---

## 2. rag —— 知识库检索

### 为什么是 BM25

向量检索要额外下 embedding 模型、装向量库，链路长。BM25 纯 CPU、毫秒级、零依赖，
而且对**专有名词、编号、术语**召回更准。本项目的定位是"轻量可部署"，不是"检索效果天花板"。

### 分词

中英混合，各走各的：

```python
WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
```

- 英文/数字：按词切，小写化（`BM25` → `bm25`，`Qwen2.5` → `qwen2.5`）
- 中文：字符 unigram + 相邻 bigram（`梯度下降` → `梯,度,下,降,梯度,度下,下降`）

**查询侧只发 bigram**（长度 ≥2 的中文片段），这是被单元测试逼出来的修正：

```python
# 修正前：查询"量子纠缠退相干"按单字切，会命中文档里"相关性"的"相"，拿到假分
# 修正后：只发 bigram，"量子/子纠/纠缠/缠退/退相/相干"都不在文档里 -> 0 分
```

代价是单字查询（"猫"）要靠长度 1 的特例兜住。

### 排序

标准 BM25，`k1=1.5`、`b=0.75`：

```
score(D,Q) = Σ IDF(q) · ( f(q,D)·(k1+1) ) / ( f(q,D) + k1·(1-b+b·|D|/avgdl) )
```

### 切片

按空行切段，再按 `size`（默认 400 字）合并，`overlap`（默认 80 字）保留上下文重叠。
标题行会跟着后面的内容一起进块，避免"孤儿标题"。

### 注入方式

检索到的片段拼成一段 context，追加到 **system 消息**：

```
以下是检索到的参考资料，请优先依据它们回答，并在答案中标注来源文件名：

[来源 bm25-notes.md]
……片段内容……

[来源 ollama-notes.md]
……片段内容……
```

### 用法

把 `.md` / `.txt` 放进 `data/docs/`，索引会在文件 mtime 变化时自动重建；也可以直接
`POST /api/knowledge {"filename": "x.md", "content": "..."}`。

---

## 3. memory —— 会话记忆

### 机制

- SQLite 单文件存 `messages(session_id, role, content, created_at)`
- 请求带 `session_id` 时，**最近 N 条历史自动拼回 messages 前面**
- 回答完自动写回本轮

```python
resp = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=[{"role": "user", "content": "我叫什么名字？"}],
    extra_body={"augmentations": ["memory"], "session_id": "user-42"},
)
```

第二轮问"我叫什么名字"，模型能答出来 —— 实测：

```
[memory] 注入条数: {'injected': 2}
[memory] 第二轮回答: 您的名字是徐悦，您正在做的项目是单词雨。
```

### 局限

**没有压缩**。对话轮次多了，历史会一直累积 token，很快吃掉上下文窗口。
生产环境该加摘要压缩或滑动窗口 —— 这是已知限制，README 里也标了。

---

## 4. guard —— 输入护栏

### 两件事

1. **PII 脱敏**（正则）

   | 类型 | 规则 | 输出 |
   |---|---|---|
   | 手机号 | `1[3-9]\d{9}` | `[REDACTED:phone]` |
   | 身份证 | `\d{17}[\dXx]` | `[REDACTED:id_card]` |
   | 邮箱 | 标准邮箱正则 | `[REDACTED:email]` |

   ```python
   redact("我的手机 13812345678，邮箱 a.b@example.com")
   # -> 我的手机 [REDACTED:phone]，邮箱 [REDACTED:email]
   ```

   刻意**不**匹配普通短数字（订单号、金额），避免误伤。

2. **长度限制**：超过 `max_input_chars`（默认 20000）直接拒绝，返回 413。

脱敏是**就地改写**发给模型的内容，不是拦下整个请求 —— 用户体验上更顺，
但也意味着日志里如果记录了原文仍会有 PII，生产环境要注意日志脱敏。

---

## 扩展：加一个新外挂

`augment/` 下的约定很简单：

1. 写一个模块，实现 `prepare(...)` 里的预处理 和 `finalize(...)` 里的收尾；
2. 在 `AVAILABLE` 里登记名字；
3. 在 `augment/__init__.py` 的 `prepare()` / `finalize()` 里各加一段。

比如想加"联网搜索"，就是在 `prepare()` 里检索网页，把结果按 RAG 的方式注入 system，
然后在 `AVAILABLE` 里加个 `"web"`。流式和非流式会自动都支持。
