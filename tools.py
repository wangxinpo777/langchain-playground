"""
工具模块：定义可供 LLM 调用的工具
"""

from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool

from config import RETRIEVER_TOP_K


def create_retriever_tool(store: Chroma):
    """创建检索工具，供模型按需调用

    Args:
        store: Chroma 向量库实例

    Returns:
        检索工具函数
    """
    retriever = store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    @tool
    def search_knowledge_base(query: str) -> str:
        """在本地知识库中搜索相关信息。

        使用场景：
        - 用户问题涉及特定文档、专业领域知识
        - 需要引用具体数据、事实、代码片段
        - 你不确定答案或需要验证信息时

        Args:
            query: 搜索查询词，应该是简洁的关键词或问题

        Returns:
            检索到的相关文档内容，如果没有找到则返回提示信息
        """
        docs = retriever.invoke(query)
        if not docs:
            return "知识库中未找到相关信息"
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    return search_knowledge_base
