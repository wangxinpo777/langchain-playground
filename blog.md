# 从零构建本地知识库问答系统：LangChain + DeepSeek + Chroma 实战

> 你是否想过，把一份 PDF、一个网页、甚至一整个文档目录丢进去，然后直接用自然语言提问，让 AI 帮你检索并回答？本文带你用不到 200 行 Python 代码实现这件事。

---

## 背景：为什么需要 RAG？

大语言模型（LLM）天生有两个缺陷：

1. **知识截止**：训练数据有截止日期，无法了解最新信息。
2. **幻觉**：模型可能一本正经地编造并不存在的答案。

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是目前业界最主流的解决方案。它的思路简洁而优雅：

```
用户问题 → 向量检索相关文档片段 → 把片段塞进 Prompt → LLM 基于真实资料回答
```

这样，模型的回答有了"参考资料"的支撑，幻觉大幅减少，还能实时更新知识库。

---

## 项目介绍

本项目是一个基于 **LangChain + DeepSeek + Chroma** 的完整 RAG 问答系统，具备以下特性：

| 特性 | 说明 |
|------|------|
| 多格式文档支持 | `.txt` / `.pdf` / `.md` 及整个目录批量加载 |
| 网页内容加载 | 直接传入 URL，自动抓取并解析 |
| 中文 Embedding | 使用 `BAAI/bge-small-zh-v1.5`，专为中文优化 |
| 持久化向量库 | Chroma 存盘，避免每次重复计算 |
| 多轮对话 | 交互模式下支持上下文连贯的连续问答 |
| 流式输出 | 回答实时逐字打印，体验更流畅 |

---

## 技术栈

- **LangChain** — 构建 LLM 应用的框架，提供文档加载、文本切分、链式调用等能力
- **DeepSeek** — 国产高性价比大语言模型，API 兼容 OpenAI 格式
- **ChromaDB** — 轻量级本地向量数据库，支持持久化
- **sentence-transformers** — 加载 HuggingFace Embedding 模型

---

## 架构设计

项目被拆分为六个职责清晰的模块：

```
main.py              ← 入口，解析参数，组合流程
config.py            ← 全局配置（模型、参数、API Key）
loader.py            ← 文档加载（文件 / 目录 / URL）
vector_store.py      ← 向量库构建与持久化
rag_chain.py         ← RAG 链：检索 → Prompt → LLM → 输出
chat_history.py      ← 多轮对话历史管理
CallbackHandler.py   ← 调试回调，打印中间过程
```

核心数据流如下：

```
[文档] → 切分 → Embedding → ChromaDB
                                ↓
[用户问题] → 向量检索 → 相关片段 → Prompt → DeepSeek → 回答
```

---

## 核心代码解析

### 1. 文档加载与切分

LangChain 提供了统一的文档加载接口，无论是本地文件还是网页，都能一行代码搞定。

```python
# 支持 txt / pdf / md 文件
loader = TextLoader(file_path, encoding="utf-8")

# 支持网页 URL
loader = WebBaseLoader(url)

# 切分成小块（中文场景建议 200-500）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)
docs = splitter.split_documents(raw_docs)
```

**为什么要切分？** 向量检索需要把文本转为固定维度的向量，太长的文本会丢失细节；切成合适大小的块，检索精度更高。

### 2. 向量库构建与复用

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 首次构建，持久化到磁盘
store = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")

# 后续直接加载，无需重新计算
store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
```

持久化的好处是显而易见的：Embedding 计算有一定耗时，把结果存到磁盘，下次启动直接复用，节省大量等待时间。

### 3. RAG 链：LCEL 声明式写法

这是整个项目最核心的部分。LangChain 的 LCEL（表达式语言）用管道符 `|` 串联各个组件：

```python
chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

读起来就像一条流水线：
1. 用问题去检索，把文档格式化为字符串作为 `context`
2. 连同 `question` 一起填入 Prompt 模板
3. 传给 DeepSeek 模型
4. 解析输出为字符串

### 4. 多轮对话支持

交互模式下，链会携带历史消息，让模型理解上下文：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个问答助手，请根据以下上下文回答...\n\n【上下文】\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),  # 注入历史消息
    ("human", "{question}"),
])
```

---

## 快速上手

### 环境准备

```bash
# 克隆项目
git clone https://github.com/wangxinpo777/langchain-playground
cd langchain-playground

# 安装依赖
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 DeepSeek API Key（在 [platform.deepseek.com](https://platform.deepseek.com) 免费注册获取）：

```
DEEPSEEK_API_KEY=your_api_key_here
```

### 运行示例

```bash
# 最简单的方式：使用内置示例文本，直接提问
python main.py

# 加载你自己的文档
python main.py --file your_doc.pdf

# 批量加载目录下所有文档
python main.py --file ./docs

# 从网页抓取内容
python main.py --url https://docs.langchain.com/docs/

# 进入交互式多轮对话模式
python main.py --interactive

# 强制重建向量库（修改文档后使用）
python main.py --rebuild
```

---

## 实际效果演示

以下是交互模式的一段对话示例（知识库为 LangChain 文档）：

```
❓ 你的问题: LangChain 是什么？

LangChain 是一个用于构建大语言模型应用的开源框架。它提供了一系列工具和
抽象层，方便开发者将 LLM 与外部数据源、工具和记忆模块结合起来...

❓ 你的问题: 它支持哪些向量数据库？

根据文档，LangChain 支持多种向量数据库，包括 Chroma、FAISS、Pinecone 等。
其中 Chroma 是本地部署的轻量级选择，Pinecone 则适合云端生产环境...

❓ 你的问题: 上面提到的 FAISS 是什么原理？

FAISS 是 Facebook AI Research 开源的高效相似度搜索库，它通过...（基于上
文理解「上面提到的」指代 FAISS）
```

可以看到，第三个问题中模型正确理解了代词指代，这就是多轮对话历史的作用。

---

## 常见问题

**Q: 首次运行很慢？**

A: 首次运行会从 HuggingFace 下载 Embedding 模型（约 100MB），完成后会缓存到本地，后续运行秒启动。网络不好的话可以提前手动下载模型文件。

**Q: 如何更换成其他 LLM？**

A: 修改 `config.py` 中的 `BASE_URL`、`MODEL_NAME` 和 `API_KEY`。由于使用了 OpenAI 兼容接口，理论上支持任何兼容该格式的模型服务（如 Qwen、GLM、本地 Ollama 等）。

**Q: PDF 中文乱码？**

A: 确保安装了 `pypdf>=4.0.0`，部分扫描版 PDF 需要额外的 OCR 工具处理。

**Q: 如何评估检索质量？**

A: 调整 `config.py` 中的 `RETRIEVER_TOP_K`（默认 3）和 `CHUNK_SIZE`（默认 200）。一般来说，chunk 越小检索越精准，但可能丢失上下文；`top_k` 越大召回越全，但 Prompt 也越长。

---

## 可扩展方向

这个项目是一个扎实的 RAG 基础实现，在此基础上你可以继续探索：

- **HyDE（假设文档嵌入）**：先让 LLM 生成一个假设答案，再用这个答案去检索，提升召回率
- **重排序（Rerank）**：检索后用交叉编码器对结果重新排序，提升精度
- **Agent 模式**：让模型自主决定什么时候检索、检索什么
- **Web UI**：接入 Streamlit 或 Gradio，做成可视化界面
- **多知识库路由**：根据问题自动选择不同的知识库

---

## 总结

这个项目用不到 300 行代码（7 个模块）实现了一个生产可用的 RAG 系统骨架：

- 模块化设计，每个文件职责单一，便于扩展
- 中文友好，Embedding 模型专为中文优化
- 开箱即用，5 分钟完成配置和首次运行
- 成本极低，DeepSeek API 价格仅为 OpenAI 的约 1/30

如果你在学习 LangChain 或想把 RAG 落地到自己的项目中，这个项目是一个很好的起点。欢迎 Star、Fork，也欢迎提 Issue 和 PR！

---

*本文代码已在 Python 3.11+ 环境测试通过。如有问题，欢迎在评论区交流。*
