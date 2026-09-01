# 游戏性能优化 AI Agent

基于 LangChain + LangGraph + Milvus 构建的垂直领域 Agent，用于游戏性能问题的智能辅助排查。

## 架构

```
User → 意图识别 → 工具路由 → [Profiler API / 静态分析 / Milvus RAG] → 结论生成 → 反馈沉淀

```

- **意图识别 & 工具路由**：由 LLM Function Calling 实现，根据问题类型分发到对应工具
- **实时分析工具**：帧率日志解析（正则）、代码复杂度扫描（AST 静态分析）
- **RAG 记忆层**：Milvus 向量库 + bge-small-zh 本地 Embedding，支持按 category 标量过滤
- **反馈闭环**：每次排查结果可一键回写 Milvus，知识库持续膨胀

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入你的 API Key
python test_rag.py     # 验证 RAG 检索
python main.py         # 启动 Agent

```

## 演示用例

| 输入 | 触发工具 |
|---|---|
| 帮我分析下 test_logs/fps.log | analyze_fps_log |
| 这段代码为什么这么慢：`for i in range(n): for j in range(n): ...` | scan_code_complexity |
| 同屏人多就卡 | query_optimization_history |

## RAG 模块

- **向量库**：Milvus（默认 Lite 本地模式，环境变量一键切集群）
- **Embedding**：BAAI/bge-small-zh-v1.5，768 维
- **索引**：COSINE + AUTOINDEX，支持 category 标量过滤（混合检索）
- **数据溯源**：source 字段区分人工录入 / Agent 沉淀 / 复盘文档