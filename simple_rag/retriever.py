"""检索器：把"向量化 + 向量库检索"两步封装成一个动作。"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# 从同级的 embeddings 模块导入两个向量化器类，仅用于类型标注。
from .embeddings import TfidfEmbedder, SentenceTransformerEmbedder
# 从同级的 vector_store 模块导入检索结果数据类与向量库类。
from .vector_store import Result, VectorStore


class Retriever:
    """给定一个问题，返回最相关的若干文本块。"""

    def __init__(self, embedder: TfidfEmbedder | SentenceTransformerEmbedder, store: VectorStore) -> None:
        # 记住"负责把文本变成向量"的对象（建库与提问必须用同一个，口径才一致）。
        self.embedder = embedder
        # 记住"存放所有向量并能算相似度"的向量库。
        self.store = store

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[Result]:
        # 第一步：把用户的问题也编码成向量。
        query_vector = self.embedder.encode(query)
        # 第二步：拿问题向量去向量库里检索，返回相似度最高的 top_k 个结果。
        return self.store.search(query_vector, top_k=top_k, min_score=min_score)

