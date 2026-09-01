import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools import performance_tools

load_dotenv()

SYSTEM_PROMPT = """你是游戏客户端性能优化专家 Agent，服务于游戏研发团队的性能问题排查。

## 你的工作流程（意图识别 → 工具路由 → 结论生成）

1. **意图识别**：判断用户问题属于哪一类 —— 帧率异常 / 内存问题 / CPU 瓶颈 / 启动卡顿 / 通用咨询
2. **工具路由**：根据意图选择工具，遵守以下优先级：
   - 涉及帧率、卡顿、日志 → 先调 analyze_fps_log
   - 涉及代码逻辑、算法耗时 → 先调 scan_code_complexity
   - 任何性能问题 → 优先调 query_optimization_history 查历史经验，避免重复排查
   - 问题已解决且用户确认 → 调 save_new_case 沉淀到知识库
3. **结论生成**：综合工具返回的真实数据 + 历史案例，给出可落地的修复建议

## 输出要求
- 先展示工具返回的原始数据（让结论可追溯）
- 再给出分析结论
- 最后给出具体修复建议，尽量精确到代码级或配置项
- 如果检索到历史案例，明确标注"参考历史案例"
"""


def build_agent():
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0,
    )
    agent = create_react_agent(
        model=llm,
        tools=performance_tools,
        prompt=SYSTEM_PROMPT,
    )
    return agent


# 全局单例
agent = build_agent()