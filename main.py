from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent_core import agent
from rag_system import rag

load_dotenv()


def main():
    print("=" * 55)
    print("  游戏性能优化 AI Agent（Milvus RAG 版）")
    print("  工具：帧率日志分析 | 代码静态扫描 | 历史案例检索")
    print(f"  知识库容量：{rag.count()} 条")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 55)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见。")
            break

        print("\nAgent 处理中...\n")
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

        # 取最后一条 AI 消息
        ai_msg = result["messages"][-1].content
        print("-" * 55)
        print(ai_msg)
        print("-" * 55)

        # 反馈闭环：询问是否沉淀
        save = input("\n是否将本次案例存入知识库？(y/N): ").strip().lower()
        if save == "y":
            category = input("分类（GPU/CPU/内存/启动优化）：").strip() or "未分类"
            rag.insert_cases([{
                "issue": user_input,
                "solution": ai_msg,
                "category": category,
                "source": "用户会话沉淀",
            }])
            print(f"✅ 已入库，当前库容量：{rag.count()} 条")


if __name__ == "__main__":
    main()