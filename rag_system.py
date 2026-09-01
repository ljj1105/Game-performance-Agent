import os
from typing import List, Dict
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

load_dotenv()


class MilvusRAG:
    """直连 Milvus 的游戏性能优化案例检索库"""

    def __init__(self):
        self.uri = os.getenv("MILVUS_URI", "./milvus_game_perf.db")
        self.collection = os.getenv("COLLECTION_NAME", "game_optimize_cases")
        self.token = os.getenv("MILVUS_TOKEN", "")

        connect_kwargs = {"uri": self.uri}
        if self.token:
            connect_kwargs["token"] = self.token
        self.client = MilvusClient(**connect_kwargs)
        print(f"[Milvus] 已连接: {self.uri}")

        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name=model_name, device="cpu"
        )
        self.dim = len(self.ef.encode_queries(["test"])[0])

        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.collection):
            print(f"[Milvus] Collection '{self.collection}' 已存在，直接加载")
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("issue", DataType.VARCHAR, max_length=2048)
        schema.add_field("solution", DataType.VARCHAR, max_length=8192)
        schema.add_field("category", DataType.VARCHAR, max_length=256)
        schema.add_field("source", DataType.VARCHAR, max_length=256)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX",
        )
        self.client.create_collection(
            self.collection, schema=schema, index_params=index_params
        )
        print(f"[Milvus] Collection 创建完成，向量维度={self.dim}")

    def insert_cases(self, cases: List[Dict[str, str]]):
        """批量写入案例（首次灌库 / 反馈闭环新增）"""
        issues = [c["issue"] for c in cases]
        vectors = self.ef.encode_documents(issues)

        data = [
            {
                "vector": vec,
                "issue": c["issue"],
                "solution": c["solution"],
                "category": c.get("category", "未分类"),
                "source": c.get("source", "人工录入"),
            }
            for vec, c in zip(vectors, cases)
        ]
        res = self.client.insert(self.collection, data=data)
        print(f"[Milvus] 写入 {res.get('insert_count', len(data))} 条案例")

    def search(self, query: str, top_k: int = 3, category: str = None) -> str:
        """语义检索 + 可选分类过滤"""
        query_vec = self.ef.encode_queries([query])

        search_params = {"metric_type": "COSINE"}
        expr = f'category == "{category}"' if category else None

        hits = self.client.search(
            self.collection,
            data=query_vec,
            limit=top_k,
            output_fields=["issue", "solution", "category", "source"],
            search_params=search_params,
            expr=expr,
        )

        if not hits or not hits[0]:
            return "未在历史案例库中检索到相似记录。"

        context = "### 检索到以下历史优化案例（按相似度排序）：\n"
        for i, hit in enumerate(hits[0], 1):
            e = hit["entity"]
            score = hit["distance"]
            context += (
                f"【案例{i}】相似度={score:.4f} | 分类={e.get('category', '-')}\n"
                f"问题：{e['issue']}\n"
                f"方案：{e['solution']}\n"
                f"来源：{e.get('source', '-')}\n---\n"
            )
        return context

    def count(self) -> int:
        return self.client.get_collection_stats(self.collection).get("row_count", 0)


# ============ 单例 + 首次灌库 ============
rag = MilvusRAG()

if rag.count() == 0:
    print("[Milvus] 知识库为空，执行首次灌库...")
    INITIAL_CASES = [
        {
            "issue": "开放世界场景加载时 FPS 骤降至 15",
            "solution": "纹理流送池过小，将 r.Streaming.PoolSize 从 500 调至 1000 后恢复。",
            "category": "内存",
            "source": "",
        },
        {
            "issue": "多人同屏释放技能时严重卡顿",
            "solution": "粒子特效 GPU 开销过大，改为 GPU Sprite 模式并限制同屏最大粒子数为 500。",
            "category": "GPU",
            "source": "战斗组技术分享",
        },
        {
            "issue": "物理碰撞检测耗时过高导致掉帧",
            "solution": "复杂 Mesh 使用了精确碰撞，替换为简化凸包（Convex Hull）后 PhysX 耗时下降 60%。",
            "category": "CPU",
            "source": "引擎组优化文档",
        },
        {
            "issue": "Shader 编译导致游戏启动卡顿",
            "solution": "启用 Allow Asynchronous Shader Compilation 并预编译常用材质，启动时间从 40s 降至 12s。",
            "category": "启动优化",
            "source": "渲染组踩坑记录",
        },
        {
            "issue": "运行 2 小时后帧率持续下降直至崩溃",
            "solution": "定位到 TextureManager 中纹理资源未在场景切换时 Release，存在内存泄漏，加回收逻辑后稳定。",
            "category": "内存",
            "source": "QA 回归缺陷",
        },
    ]
    rag.insert_cases(INITIAL_CASES)