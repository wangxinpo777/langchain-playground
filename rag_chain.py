"""
RAG 链构建模块：组装检索 → Prompt → LLM → 输出解析的完整链
"""

from pydantic import SecretStr

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI

from config import (
    API_KEY,
    BASE_URL,
    MODEL_NAME,
    RETRIEVER_TOP_K,
    TEMPERATURE,
)


def format_docs(docs: list[Document]) -> str:
    """把多个 Document 的正文拼接成一个字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(store: Chroma, enable_history: bool = False):
    """组装完整的 RAG 链：检索 → 填充 Prompt → 调用 LLM → 解析输出

    Args:
        store: Chroma 向量库
        enable_history: 是否启用对话历史（交互模式下为 True）
    """

    retriever = store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    # 根据是否启用历史选择不同的提示词模板
    if enable_history:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个问答助手，请根据以下检索到的上下文内容回答问题。
如果上下文中没有足够信息，请如实说"我不知道"，不要编造答案。

【上下文】
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
    else:
        prompt = ChatPromptTemplate.from_template("""你是一个问答助手，请根据以下检索到的上下文内容回答问题。
如果上下文中没有足够信息，请如实说"我不知道"，不要编造答案。

【上下文】
{context}

【问题】
{question}
""")

    llm = ChatOpenAI(
        api_key=SecretStr(API_KEY),
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        streaming=True,
    )

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            **({"chat_history": lambda x: x.get("chat_history", [])} if enable_history else {}),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
