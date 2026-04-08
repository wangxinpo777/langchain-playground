"""
全局配置常量
"""

# DeepSeek API 配置
API_KEY = "sk-38edce4f0bea49fca058e32cedb96886"
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
TEMPERATURE = 0

# 文本切分参数
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# 检索参数
RETRIEVER_TOP_K = 3

# Embedding 模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 提示词模板
PROMPT_TEMPLATE = """\
你是一个问答助手，请根据以下检索到的上下文内容回答问题。
如果上下文中没有足够信息，请如实说"我不知道"，不要编造答案。

【上下文】
{context}

【问题】
{question}
"""

# 内置示例文本
SAMPLE_TEXT = """
LangChain 是一个用于构建大语言模型应用的开源框架。
它提供了一系列工具和抽象层，方便开发者将 LLM 与外部数据源、工具和记忆模块结合起来。

RAG（Retrieval-Augmented Generation，检索增强生成）是一种常见的 LLM 应用模式。
它的核心思路是：先从外部知识库检索与问题相关的文档片段，再把这些片段作为上下文
传给 LLM，让模型基于真实知识来生成答案，从而减少"幻觉"。

LangChain 支持多种向量数据库，如 Chroma、FAISS、Pinecone 等。
InMemoryVectorStore 是一个轻量级的内存向量库，适合快速原型开发。

LangChain Expression Language（LCEL）是 LangChain 的声明式链式编程接口。
通过管道符 | 可以将 Prompt、LLM、输出解析器等组件串联成一条可运行的链。
"""
