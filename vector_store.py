"""
向量库构建模块：切分文档、向量化并存入内存向量库
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


def build_vector_store(docs: list[Document]) -> InMemoryVectorStore:
    """把文档切分成小块，向量化后存入内存向量库，返回向量库对象"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[切分] 共 {len(chunks)} 个文本块")

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = InMemoryVectorStore(embedding=embedding)
    store.add_documents(chunks)
    return store
