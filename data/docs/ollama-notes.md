# Ollama 与本地模型服务笔记

## Ollama 是什么

Ollama 是一个把大模型权重、推理引擎（llama.cpp / ggml）和 HTTP 服务打包在一起的运行时。
安装后执行 `ollama pull qwen2.5:1.5b` 会把模型拉到本地 `~/.ollama/models`，然后用
`ollama serve` 在 11434 端口起一个 HTTP 服务。

它默认只监听 `127.0.0.1`，如果设置环境变量 `OLLAMA_HOST=0.0.0.0`，就会变成所有网卡都监听——
**注意这时候是没有任何鉴权的**，任何能连到这个端口的人都可以白嫖你的显卡和模型，所以
对外暴露之前必须自己加一层网关。

## Ollama 的两种接口

- `/api/generate`：单轮补全，给一段 prompt 拿一段文本。
- `/api/chat`：多轮对话，输入 messages 数组，支持 tools（function calling）和 format（JSON Schema 约束解码）。

此外 Ollama 还自带一个 OpenAI 兼容层 `/v1/chat/completions`，方便现成的客户端直接接。

## 参数字段的坑

Ollama 把采样参数放在 `options` 里，而且名字跟 OpenAI 不一样：

- `num_predict` 对应 OpenAI 的 `max_tokens`（最大生成 token 数）
- `temperature`、`top_p`、`seed`、`stop` 名字一致，但同样在 `options` 里
- 上下文窗口是 `num_ctx`，默认值比模型本身的支持长度小，长文本要手动调大

## 为什么要做网关

1. **鉴权与限流**：Ollama 本身没有这两样东西。
2. **协议统一**：不同客户端说不同的话（OpenAI 风格、Anthropic 风格），网关负责翻译。
3. **能力增强**：工具调用、知识库检索、会话记忆这些逻辑放在网关，客户端就轻了。
4. **可观测**：统计每个协议的调用量、token 消耗和延迟。
