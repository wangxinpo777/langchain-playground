"""
RAG 链构建模块：组装检索 → Prompt → LLM → 输出解析的完整链
"""

from pydantic import SecretStr

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI

from config import (
    API_KEY,
    BASE_URL,
    MODEL_NAME,
    PROMPT_TEMPLATE,
    RETRIEVER_TOP_K,
    TEMPERATURE,
)


def format_docs(docs: list[Document]) -> str:
    """把多个 Document 的正文拼接成一个字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(store: InMemoryVectorStore):
    """组装完整的 RAG 链：检索 → 填充 Prompt → 调用 LLM → 解析输出"""

    retriever = store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatOpenAI(
        api_key=SecretStr(API_KEY),
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        streaming=True,
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
