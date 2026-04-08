from langchain_core.callbacks import BaseCallbackHandler


class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.answer_started = False

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.answer_started = False  # 每次调用重置

        print("\n" + "=" * 60)
        print("【完整提示词】")
        print("=" * 60)
        print(prompts[0])
        print("=" * 60 + "\n")

    def on_llm_new_token(self, token, **kwargs):
        if not self.answer_started:
            print("【回答】", flush=True)
            self.answer_started = True

        print(token, end="", flush=True)

    def on_retriever_end(self, documents, **kwargs):
        print("\n🔍 [检索到的文档片段]")
        for i, doc in enumerate(documents, 1):
            print(f"\n--- 文档 {i} ---")
            print(doc.page_content[:200])  # 防止太长
