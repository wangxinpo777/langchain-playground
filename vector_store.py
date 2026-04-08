"""
向量库构建模块：切分文档、向量化并存入 Chroma 持久化向量库
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, CHROMA_DIR


def build_vector_store(docs: list[Document], force_rebuild: bool = False) -> Chroma:
    """把文档切分成小块，向量化后存入 Chroma 持久化向量库，返回向量库对象

    Args:
        docs: 文档列表
        force_rebuild: 是否强制重建向量库（默认 False，如果已存在则直接加载）
    """

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 如果向量库已存在且不强制重建，直接加载
    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        print(f"[加载] 从 {CHROMA_DIR} 加载已有向量库")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    # 否则重新构建
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[切分] 共 {len(chunks)} 个文本块")

    print(f"[构建] 创建向量库并保存到 {CHROMA_DIR}")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    return store
