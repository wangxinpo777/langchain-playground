"""
对话历史管理模块：支持多轮问答的上下文记忆
"""

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class InMemoryChatHistory:
    """内存中的对话历史存储"""

    def __init__(self):
        self.store = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取指定会话的历史记录"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]


def build_conversational_chain(base_chain):
    """将基础 RAG 链包装为支持对话历史的链

    Args:
        base_chain: 基础的 RAG 链（不含历史记忆）

    Returns:
        支持对话历史的链
    """
    history_manager = InMemoryChatHistory()

    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        history_manager.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return conversational_chain
