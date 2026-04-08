"""
RAG Demo — LangChain + DeepSeek + Chroma
用法:
  python main.py                              # 使用内置示例文本
  python main.py --file docs.txt              # 从本地文件加载
  python main.py --file ./docs                # 从目录批量加载
  python main.py --url https://...            # 从网页 URL 加载
  python main.py --rebuild                    # 强制重建向量库
  python main.py --interactive                # 交互式问答模式
"""

import argparse

from loader import load_documents
from vector_store import build_vector_store
from rag_chain import build_rag_chain
from chat_history import build_conversational_chain
from callback_handler import DebugCallbackHandler


def interactive_mode(chain):
    """交互式问答模式：持续接收用户输入并回答，支持多轮对话"""
    print("\n" + "=" * 60)
    print("🤖 进入交互式问答模式（支持多轮对话）")
    print("=" * 60)
    print("提示：输入问题后按回车，输入 'exit' 或 'quit' 退出\n")

    session_id = "default_session"

    while True:
        try:
            question = input("❓ 你的问题: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "退出"]:
                print("\n👋 再见！")
                break

            print()
            chain.invoke(
                {"input": question},
                config={
                    "configurable": {"session_id": session_id},
                    "callbacks": [DebugCallbackHandler()]
                }
            )
            print("\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", help="从本地文件或目录加载文档（支持 .txt/.pdf/.md）")
    group.add_argument("--url", help="从网页 URL 加载文档")
    parser.add_argument(
        "--question",
        "-q",
        default="LangChain 是什么？它支持哪些功能？",
        help="要提问的问题（默认使用示例问题）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建向量库（默认会复用已有向量库）",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="进入交互式问答模式（支持多轮对话）",
    )
    args = parser.parse_args()

    try:
        docs = load_documents(args)
        store = build_vector_store(docs, force_rebuild=args.rebuild)

        if args.interactive:
            # 交互模式：启用对话历史
            base_chain = build_rag_chain(store, enable_history=True)
            chain = build_conversational_chain(base_chain)
            interactive_mode(chain)
        else:
            # 单次问答：不启用对话历史
            chain = build_rag_chain(store, enable_history=False)
            chain.invoke({"input": args.question}, config={"callbacks": [DebugCallbackHandler()]})
            print()

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
