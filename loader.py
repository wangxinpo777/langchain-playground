"""
文档加载模块：根据命令行参数从不同来源加载文档
支持：本地文件（.txt/.pdf/.md）、目录批量加载、网页 URL、内置示例
"""

import argparse
import os
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

from config import SAMPLE_TEXT


def load_documents(args: argparse.Namespace) -> list[Document]:
    """根据命令行参数选择文档来源，返回 Document 列表"""

    try:
        # 本地文件或目录
        if args.file:
            path = Path(args.file)

            if not path.exists():
                raise FileNotFoundError(f"路径不存在: {args.file}")

            # 目录批量加载
            if path.is_dir():
                print(f"[文档来源] 目录: {args.file}")
                docs = []

                # 加载 .txt 文件
                txt_loader = DirectoryLoader(
                    args.file, glob="**/*.txt", loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                docs.extend(txt_loader.load())

                # 加载 .md 文件
                md_loader = DirectoryLoader(
                    args.file, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
                )
                docs.extend(md_loader.load())

                # 加载 .pdf 文件
                pdf_loader = DirectoryLoader(
                    args.file, glob="**/*.pdf", loader_cls=PyPDFLoader
                )
                docs.extend(pdf_loader.load())

                if not docs:
                    raise ValueError(f"目录 {args.file} 中未找到支持的文档（.txt/.md/.pdf）")

                print(f"[加载] 共加载 {len(docs)} 个文档")
                return docs

            # 单个文件
            ext = path.suffix.lower()
            print(f"[文档来源] 本地文件: {args.file}")

            if ext == ".txt":
                loader = TextLoader(args.file, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(args.file)
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(args.file)
            else:
                raise ValueError(f"不支持的文件格式: {ext}，仅支持 .txt/.pdf/.md")

            return loader.load()

        # 网页 URL
        if args.url:
            print(f"[文档来源] 网页 URL: {args.url}")
            loader = WebBaseLoader(args.url)
            return loader.load()

        # 内置示例
        print("[文档来源] 内置示例文本")
        return [Document(page_content=SAMPLE_TEXT)]

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        raise
    except ValueError as e:
        print(f"❌ 错误: {e}")
        raise
    except Exception as e:
        print(f"❌ 加载文档时出错: {e}")
        raise
