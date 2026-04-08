"""
文档加载模块：根据命令行参数从不同来源加载文档
"""

import argparse

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_core.documents import Document

from config import SAMPLE_TEXT


def load_documents(args: argparse.Namespace) -> list[Document]:
    """根据命令行参数选择文档来源，返回 Document 列表"""

    if args.file:
        print(f"[文档来源] 本地文件: {args.file}")
        loader = TextLoader(args.file, encoding="utf-8")
        return loader.load()

    if args.url:
        print(f"[文档来源] 网页 URL: {args.url}")
        loader = WebBaseLoader(args.url)
        return loader.load()

    print("[文档来源] 内置示例文本")
    return [Document(page_content=SAMPLE_TEXT)]
