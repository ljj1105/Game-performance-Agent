import os
import re
import ast

from langchain.tools import tool
from rag_system import rag


# ============ 工具 1：帧率日志分析 ============
@tool
def analyze_fps_log(log_content: str) -> str:
    """
    分析游戏帧率日志，识别掉帧区间与性能瓶颈。
    Args:
        log_content: 日志文件路径，或直接粘贴的日志文本。
    """
    if os.path.exists(log_content):
        with open(log_content, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = log_content

    matches = re.findall(r'FPS[:\s]*(\d+)', content, re.IGNORECASE)
    if not matches:
        return "未检测到有效的 FPS 数据，请确保日志包含 'FPS: 60' 格式。"

    fps_values = [int(x) for x in matches]
    avg_fps = sum(fps_values) / len(fps_values)
    drop_indices = [i for i, fps in enumerate(fps_values) if fps < 30]

    report = (
        f"--- 性能日志分析报告 ---\n"
        f"总采样帧数: {len(fps_values)}\n"
        f"平均帧率: {avg_fps:.2f} FPS\n"
        f"最低帧率: {min(fps_values)} FPS\n"
    )
    if drop_indices:
        report += (
            f"\n[警告] 检测到 {len(drop_indices)} 帧低于 30FPS，"
            f"集中在第 {drop_indices[0]}~{drop_indices[-1]} 帧区间，"
            f"建议重点排查该时段的渲染负载与资源加载。"
        )
    else:
        report += "\n[通过] 帧率稳定，未发现明显卡顿。"
    return report


# ============ 工具 2：代码静态分析 ============
def _max_nesting_depth(node: ast.AST) -> int:
    depth = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
            depth = max(depth, 1 + _max_nesting_depth(child))
        else:
            depth = max(depth, _max_nesting_depth(child))
    return depth


def _has_inner_loop(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.For, ast.While, ast.ListComp, ast.SetComp,
                              ast.DictComp, ast.GeneratorExp)):
            return True
    return False


@tool
def scan_code_complexity(code_snippet: str) -> str:
    """
    静态分析代码复杂度，定位可能导致 CPU 耗时过高的逻辑缺陷。
    Args:
        code_snippet: 待分析的代码片段。
    """
    try:
        tree = ast.parse(code_snippet)
    except SyntaxError as e:
        return f"代码语法错误，无法分析: {e}"

    func_count = 0
    loop_count = 0
    nested_loop_risks = 0
    has_print = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_count += 1
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1
            if _has_inner_loop(node):
                nested_loop_risks += 1
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'print':
            has_print = True

    max_depth = _max_nesting_depth(tree)

    report = (
        f"--- 代码静态分析报告 ---\n"
        f"函数数量: {func_count}\n"
        f"循环结构数量: {loop_count}\n"
        f"最大控制流嵌套深度: {max_depth}\n"
        f"嵌套循环（O(N²) 风险点）: {nested_loop_risks} 处\n"
    )

    suggestions = []
    if nested_loop_risks > 0:
        suggestions.append(f"⚠️ 发现 {nested_loop_risks} 处嵌套循环，建议改用空间换时间或提前剪枝。")
    if max_depth > 4:
        suggestions.append("⚠️ 嵌套过深，建议拆分子函数降低认知与分支开销。")
    if has_print:
        suggestions.append("⚠️ 生产代码中存在 print 调用，会阻塞 IO 线程，建议替换为日志系统。")

    report += "\n优化建议:\n" + "\n".join(suggestions) if suggestions else "\n代码结构健康，未发现明显风险。"
    return report


# ============ 工具 3：Milvus 历史案例检索 ============
@tool
def query_optimization_history(query: str) -> str:
    """
    从 Milvus 向量库检索历史相似的性能优化案例。
    遇到性能问题时优先调用，避免重复排查。
    Args:
        query: 当前性能问题的自然语言描述。
    """
    try:
        return rag.search(query, top_k=3)
    except Exception as e:
        return f"检索服务异常: {e}，请检查 Milvus 连接状态。"


# ============ 工具 4：反馈闭环写入 ============
@tool
def save_new_case(issue: str, solution: str, category: str) -> str:
    """
    将本次成功解决的案例写入 Milvus 知识库，供后续检索复用。
    Args:
        issue: 问题描述
        solution: 最终解决方案
        category: 分类，取值 GPU / CPU / 内存 / 启动优化
    """
    rag.insert_cases([{
        "issue": issue,
        "solution": solution,
        "category": category,
        "source": "Agent 自动沉淀",
    }])
    return f"✅ 新案例已入库，当前库容量：{rag.count()} 条"


# 供 Agent 绑定的工具列表
performance_tools = [
    analyze_fps_log,
    scan_code_complexity,
    query_optimization_history,
    save_new_case,
]