# RAG Demo

基于 LangChain + DeepSeek + Chroma 的检索增强生成（RAG）问答系统。

## 功能特性

- ✅ **API Key 安全管理** — 使用 `.env` 文件管理敏感信息
- ✅ **中文优化** — 使用 `BAAI/bge-small-zh-v1.5` 中文 Embedding 模型
- ✅ **持久化向量库** — Chroma 向量库保存到磁盘，避免重复计算
- ✅ **多格式文档加载** — 支持 `.txt`、`.pdf`、`.md` 文件及目录批量加载
- ✅ **多轮对话** — 交互模式下支持上下文连贯的多轮问答
- ✅ **错误处理** — 完善的异常捕获和友好提示
- ✅ **交互式问答** — REPL 模式，无需重启程序
- ✅ **自主检索决策** — 基于 AgentExecutor 的工具调用架构，模型自主判断是否需要检索知识库，自动执行检索并整合结果

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 运行示例

```bash
# 使用内置示例文本
python main.py

# 从本地文件加载
python main.py --file docs.txt

# 从目录批量加载（支持 .txt/.pdf/.md）
python main.py --file ./docs

# 从网页加载
python main.py --url https://example.com

# 自定义问题
python main.py -q "什么是 RAG？"

# 交互式问答模式（支持多轮对话）
python main.py --interactive

# 强制重建向量库
python main.py --rebuild
```

## 项目结构

```text
.
├── main.py              # 入口文件
├── config.py            # 全局配置
├── loader.py            # 文档加载模块
├── vector_store.py      # 向量库构建模块
├── rag_chain.py         # Agent 构建模块（AgentExecutor）
├── chat_history.py      # 对话历史管理模块
├── tools.py             # 检索工具模块
├── callback_handler.py  # 调试回调处理器
├── requirements.txt     # 依赖列表
├── .env                 # 环境变量（不提交到 Git）
├── .env.example         # 环境变量模板
└── chroma_db/           # Chroma 向量库持久化目录（自动生成）
```

## 配置说明

在 `config.py` 中可以调整以下参数：

- `CHUNK_SIZE` — 文本切分块大小（默认 200）
- `CHUNK_OVERLAP` — 切分块重叠大小（默认 40）
- `RETRIEVER_TOP_K` — 检索返回的文档数量（默认 3）
- `EMBEDDING_MODEL` — Embedding 模型（默认 `BAAI/bge-small-zh-v1.5`）
- `CHROMA_DIR` — Chroma 向量库保存路径（默认 `./chroma_db`）

## 常见问题

### 首次运行很慢？

首次运行会下载 Embedding 模型（约 100MB），后续运行会直接使用缓存。

### 如何清空向量库？

删除 `chroma_db/` 目录，或使用 `--rebuild` 参数强制重建。

### 如何更换 Embedding 模型？

修改 `config.py` 中的 `EMBEDDING_MODEL`，然后使用 `--rebuild` 重建向量库。

## 许可证

MIT
