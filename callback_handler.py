"""
回调处理器模块：用于调试和显示 LLM 执行过程
"""

from langchain_core.callbacks import BaseCallbackHandler


class DebugCallbackHandler(BaseCallbackHandler):
    """调试回调处理器：显示提示词和流式输出"""

    def __init__(self):
        self.answer_started = False

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始执行时的回调"""
        self.answer_started = False  # 每次调用重置

        print("\n" + "=" * 60)
        print("【完整提示词】")
        print("=" * 60)
        print(prompts[0])
        print("=" * 60 + "\n")

    def on_llm_new_token(self, token, **kwargs):
        """LLM 生成新 token 时的回调（流式输出）"""
        if not self.answer_started:
            print("【回答】", flush=True)
            self.answer_started = True

        print(token, end="", flush=True)

    def on_retriever_end(self, documents, **kwargs):
        """检索器完成检索时的回调"""
        print("\n🔍 [检索到的文档片段]")
        for i, doc in enumerate(documents, 1):
            print(f"\n--- 文档 {i} ---")
            print(doc.page_content[:200])  # 防止太长
