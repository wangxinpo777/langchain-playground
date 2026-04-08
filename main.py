"""
RAG Demo — LangChain + OpenAI + InMemoryVectorStore
用法:
  python main.py                        # 使用内置示例文本
  python main.py --file docs.txt        # 从本地文本文件加载
  python main.py --url https://...      # 从网页 URL 加载
"""

import argparse

from loader import load_documents
from vector_store import build_vector_store
from rag_chain import build_rag_chain
from CallbackHandler import DebugCallbackHandler


def main():
    parser = argparse.ArgumentParser(description="RAG Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", help="从本地 .txt 文件加载文档")
    group.add_argument("--url", help="从网页 URL 加载文档")
    parser.add_argument(
        "--question",
        "-q",
        default="LangChain 是什么？它支持哪些功能？",
        help="要提问的问题（默认使用示例问题）",
    )
    args = parser.parse_args()

    docs = load_documents(args)
    store = build_vector_store(docs)
    chain = build_rag_chain(store)
    chain.invoke(args.question, config={"callbacks": [DebugCallbackHandler()]})
    print()


if __name__ == "__main__":
    main()
