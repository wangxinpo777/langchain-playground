"""
RAG 链构建模块：支持模型自主决定是否检索的智能 RAG 架构
"""

from pydantic import SecretStr

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from langchain_openai import ChatOpenAI

from config import (
    API_KEY,
    BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
)
from tools import create_retriever_tool

_SYSTEM_PROMPT = """你是一个智能助手，可以回答各种问题。

你有一个本地知识库检索工具 search_knowledge_base，可以在需要时使用。

使用策略：
- 对于常识性问题、通用知识，你可以直接回答
- 对于涉及特定文档、专业领域、具体数据的问题，使用检索工具
- 如果不确定答案的准确性，使用检索工具验证
- 检索后如果没有找到相关信息，你可以基于自己的知识给出合理回答，但要说明这不是来自知识库

回答要求：
- 准确、简洁、有帮助
- 如果使用了检索结果，基于检索内容回答
- 如果检索无果但你有相关知识，可以回答并说明信息来源"""


def build_rag_chain(store: Chroma, enable_history: bool = False):
    """构建支持自主检索的智能 RAG Agent

    使用 AgentExecutor 驱动工具调用循环，模型可以：
    1. 自主决定是否需要检索知识库
    2. 执行检索并将结果纳入回答
    3. 在知识库无结果时基于自身知识回答

    Args:
        store: Chroma 向量库
        enable_history: 是否启用对话历史（交互模式下为 True）
    """
    retriever_tool = create_retriever_tool(store)

    if enable_history:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    llm = ChatOpenAI(
        api_key=SecretStr(API_KEY),
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        streaming=True,
    )

    agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=False)
