# 四协议逐字段详解

这份文档把四个协议的请求/响应结构逐字段拆开，方便对照着写客户端。

所有示例的**响应体都是本项目真实跑出来的**（模型 `qwen2.5:1.5b`），不是手写编的。

---

## 0. 一张图看清结构差异

```
OpenAI Chat          OpenAI Responses        Anthropic Messages       Ollama
─────────────────────────────────────────────────────────────────────────────
messages[]           input[]                 messages[]              messages[]
  └ role/content       └ role/content          └ role/content           └ role/content
                                                          system (顶层)
                                                          max_tokens (必填)
                                                          tools[].input_schema
─────────────────────────────────────────────────────────────────────────────
choices[0]           output[]                content[]               message
  └ message            └ content[]              └ text                  └ content
    └ content            └ output_text              └ tool_use            └ tool_calls
  └ finish_reason     status                  stop_reason             done_reason
usage.prompt_tokens  usage.input_tokens      usage.input_tokens      prompt_eval_count
```

---

## 1. Ollama 原生 —— `POST /api/chat`

### 请求

```json
{
  "model": "qwen2.5:1.5b",
  "messages": [{"role": "user", "content": "现在几点了？"}],
  "stream": false,
  "options": {
    "temperature": 0.4,
    "num_predict": 200,
    "top_p": 0.9,
    "stop": ["</s>"]
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_time",
        "description": "获取当前日期、时间与星期",
        "parameters": {"type": "object", "properties": {"tz_offset_hours": {"type": "number"}}}
      }
    }
  ],
  "format": { "type": "object", "properties": {"answer": {"type": "string"}} }
}
```

**要点**

- 采样参数全在 `options` 嵌套里，名字和 OpenAI 不同：`num_predict` = `max_tokens`。
- `format` 可以直接传 JSON Schema，Ollama 会做**约束解码**，保证输出一定是合法 JSON —— 这是它比
  "提示词里写请输出 JSON"可靠得多的地方。
- 没有 `max_tokens` 顶层字段。

### 响应

```json
{
  "model": "qwen2.5:1.5b",
  "created_at": "2026-10-27T10:56:24.000000Z",
  "message": {"role": "assistant", "content": "API是应用程序编程接口……"},
  "done": true,
  "done_reason": "stop",
  "total_duration": 13011800000,
  "prompt_eval_count": 34,
  "eval_count": 16
}
```

**要点**

- `total_duration` 等时间字段单位是**纳秒**。
- token 字段叫 `prompt_eval_count` / `eval_count`，不是 `usage.prompt_tokens`。
- `done_reason` 只有 `stop` / `length`，没有 `tool_calls`（工具调用靠 `message.tool_calls` 判断）。

### 流式：NDJSON

**不是 SSE**。一行一个 JSON，没有 `data:` 前缀，没有 `[DONE]` 收尾，靠 `done` 字段判断结束：

```json
{"model":"qwen2.5:1.5b","created_at":"...","message":{"role":"assistant","content":"数"},"done":false}
{"model":"qwen2.5:1.5b","created_at":"...","message":{"role":"assistant","content":"到"},"done":false}
{"model":"qwen2.5:1.5b","created_at":"...","message":{"role":"assistant","content":"五"},"done":true,"done_reason":"stop"}
```

> 写通用流式客户端时，这一条最容易踩坑。

---

## 2. OpenAI Chat Completions —— `POST /v1/chat/completions`

### 请求

```json
{
  "model": "qwen2.5:1.5b",
  "messages": [
    {"role": "system", "content": "你是一个简洁的助手"},
    {"role": "user", "content": "现在几点了？"}
  ],
  "max_tokens": 200,
  "temperature": 0.4,
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_time",
        "description": "获取当前日期、时间与星期",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ]
}
```

### 响应

```json
{
  "id": "chatcmpl-f46e619131224c02a9efd659",
  "object": "chat.completion",
  "created": 1793098595,
  "model": "qwen2.5:1.5b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "现在是下午六点五十七分，今天是星期二。"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 34, "completion_tokens": 19, "total_tokens": 53},
  "system_fingerprint": "ollama-gateway"
}
```

**要点**

- 生态最广：LangChain、LlamaIndex、各种 Agent 框架默认都支持这个格式。
- 工具调用的 `arguments` 是 **JSON 字符串**，不是对象：
  `{"function":{"name":"f","arguments":"{\"a\":1}"}}` —— 需要自己 `json.loads`。
- 鉴权走 `Authorization: Bearer <key>`。

### 流式：SSE

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1793098661,"model":"qwen2.5:1.5b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1793098661,"model":"qwen2.5:1.5b","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1793098661,"model":"qwen2.5:1.5b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

三段式：首帧带 `role` → 中间帧只带 `delta.content` → 末帧 `finish_reason` → `[DONE]`。

---

## 3. OpenAI Responses —— `POST /v1/responses`

### 请求

```json
{
  "model": "qwen2.5:1.5b",
  "instructions": "你是一个简洁的助手",
  "input": "现在几点了？",
  "max_output_tokens": 200,
  "temperature": 0.4,
  "stream": false
}
```

`input` 可以是字符串，也可以是数组：

```json
{
  "input": [
    {"role": "user", "content": [{"type": "input_text", "text": "查一下时间"}]},
    {"type": "function_call_output", "call_id": "call_9", "output": "2026-01-01 10:00"}
  ]
}
```

**要点**

- `instructions` 是**顶层字段**，不在 `input` 里，也不叫 system message。
- 长度参数叫 `max_output_tokens`，不是 `max_tokens`。
- 工具定义是**扁平**的：`tools[].parameters`，而不是 `tools[].function.parameters`。

### 响应

```json
{
  "id": "resp_a8074c4898724be781d56d1c",
  "object": "response",
  "created_at": 1793098596,
  "status": "completed",
  "incomplete_details": null,
  "model": "qwen2.5:1.5b",
  "output": [
    {
      "type": "message",
      "id": "msg_cea8c6180b3c43a59257be49",
      "status": "completed",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "API是应用程序编程接口……", "annotations": []}]
    }
  ],
  "parallel_tool_calls": true,
  "tool_choice": "auto",
  "tools": [],
  "usage": {
    "input_tokens": 23,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 13,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 36
  },
  "error": null,
  "metadata": {}
}
```

**要点**

- 输出是**扁平数组** `output[]`，每项有自己的 `type`（`message` / `function_call` / `reasoning`），
  未来加新类型不用改结构。文本还要再往里钻两层：`output[].content[].text`。
- 结束状态叫 `status`：`completed` / `incomplete` / `failed`。
- token 字段叫 `input_tokens` / `output_tokens`。

### 流式：带事件名的 SSE

```
event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_...","object":"response","status":"in_progress","model":"qwen2.5:1.5b"}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":0,"response":{"id":"resp_...","status":"in_progress"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_...","output_index":0,"content_index":0,"delta":"数"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_...","status":"completed", "...":"..."}}
```

事件名和 `data.type` 一一对应，可以按事件类型分别处理。

---

## 4. Anthropic Messages —— `POST /v1/messages`

### 请求

```json
{
  "model": "qwen2.5:1.5b",
  "system": "你是一个简洁的助手",
  "messages": [{"role": "user", "content": "现在几点了？"}],
  "max_tokens": 200,
  "temperature": 0.4,
  "stream": false,
  "tools": [
    {
      "name": "get_current_time",
      "description": "获取当前日期、时间与星期",
      "input_schema": {"type": "object", "properties": {}}
    }
  ]
}
```

**要点（这一家最容易接错）**

1. **`max_tokens` 必填**，缺了直接 400。本网关保持了同样的行为：
   ```json
   {"error":{"type":"invalid_request_error","message":"max_tokens: Field required（Anthropic 协议中该字段必填）"}}
   ```
2. `system` 是**顶层独立字段**，不是 messages 里的一条。
3. 工具定义用 `input_schema`，不是 `parameters`。
4. **工具结果要放在 user 消息里**，不是单独的 `role=tool`：
   ```json
   {
     "role": "user",
     "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18:57"}]
   }
   ```
5. 鉴权是 `x-api-key: <key>` + 必填的 `anthropic-version: 2023-06-01`。

### 响应

```json
{
  "id": "msg_3542a8fd76494381a16c9d09",
  "type": "message",
  "role": "assistant",
  "model": "qwen2.5:1.5b",
  "content": [{"type": "text", "text": "现在是下午六点五十七分，今天是星期二。"}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 23, "output_tokens": 18}
}
```

**要点**

- `content` 是 **block 数组**，文本在 `content[0].text`。
- `stop_reason` 取值：`end_turn` / `max_tokens` / `tool_use` / `stop_sequence`。
- 工具调用也是 content 里的一项：
  ```json
  {"type": "tool_use", "id": "toolu_1", "name": "get_current_time", "input": {}}
  ```

### 流式：六种事件

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","model":"...","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"数"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":23,"output_tokens":18}}

event: message_stop
data: {"type":"message_stop"}
```

文本增量是 `delta.type == "text_delta"`；工具参数的增量是 `delta.type == "input_json_delta"`，
字段叫 `partial_json`（**增量片段**，不是完整 JSON）。

---

## 5. 选型建议

| 场景 | 用哪个 |
|---|---|
| 已有代码是 OpenAI 风格 | `/v1/chat/completions`，改动最小 |
| 想用最新的 Agent 框架 / 深度研究类应用 | `/v1/responses`，事件语义更完整 |
| 已有代码用 Claude SDK | `/v1/messages`，直接换 base_url |
| 自己写客户端、要完全控制 | `/api/chat`，最简单，字段最少 |

另外，**本网关自己的内部接口也是 `/v1/chat/completions` 形状**，
所以用 OpenAI SDK 换 `base_url` 就能接，不用改业务代码。
