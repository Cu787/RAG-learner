"""向量库 + 相似度检索。

这是 RAG 里最关键的一步：给定一个问题向量，从所有文本块向量里，
找出最相似的 Top-K 个。

相似度用"余弦相似度"：两个向量夹角越小，方向越一致，就越相似：
    cos(a, b) = (a · b) / (|a| * |b|)
取值范围 [-1, 1]，越接近 1 越相似。

（因为我们已经在 encode() 里把向量做了 L2 归一化，所以 |a|=|b|=1，
  此时余弦相似度恰好等于点积 a·b。下面的实现仍然显式除以模长，
  这样即使你换成没归一化的向量也能正确工作。）
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# dataclass：自动生成数据类的构造函数。
from dataclasses import dataclass

# 数值计算库 numpy，用它把向量当作矩阵批量运算。
import numpy as np


# 自动生成 Result(text=..., source=..., score=...) 这样的构造函数。
@dataclass
class Result:
    """一条检索结果。"""

    text: str      # 命中的文本片段
    source: str    # 该片段来自哪个文件
    score: float   # 与问题的余弦相似度，越大越相关


class VectorStore:
    """最朴素的向量库：把所有向量堆成一个矩阵，检索时算一遍相似度。

    真实项目里会换成 FAISS / Chroma / Milvus 等，但原理完全一样。
    """

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None  # 所有向量组成的矩阵，形状 (N, D)，初始为空
        self.texts: list[str] = []              # 与矩阵每一行一一对应的文本片段
        self.sources: list[str] = []            # 与矩阵每一行一一对应的来源文件名

    def add(self, vectors: np.ndarray, texts: list[str], sources: list[str]) -> None:
        # 把输入统一成 float32 的 numpy 数组（更省内存、算得更快）。
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors is None:            # 第一次添加时
            self.vectors = vectors          # 直接赋值
        else:                               # 之后再添加时
            self.vectors = np.vstack([self.vectors, vectors])  # 在竖直方向"摞起来"拼成大矩阵
        self.texts.extend(texts)            # 同步追加文本片段
        self.sources.extend(sources)        # 同步追加来源文件名

    def search(self, query_vector: np.ndarray, top_k: int = 3, min_score: float = 0.0) -> list[Result]:
        if self.vectors is None or not self.texts:  # 库里还没有任何数据
            return []                                # 直接返回空列表

        query = np.asarray(query_vector, dtype=np.float32)   # 把问题向量转成 numpy 数组
        query_norm = float(np.linalg.norm(query)) or 1.0     # 问题向量的模长；为 0 时用 1 兜底防除零

        # 计算矩阵每一行（即每个块向量）的模长。
        row_norms = np.linalg.norm(self.vectors, axis=1)
        row_norms[row_norms == 0] = 1.0                      # 把 0 模长替换成 1，避免后面除零

        # 一次矩阵乘法算出所有块与问题的相似度：点积 / 两边模长 = 余弦相似度。
        scores = (self.vectors @ query) / (row_norms * query_norm)

        # np.argsort 默认从小到大排序；加负号就变成从大到小（高分排前面）。
        order = np.argsort(-scores)

        results: list[Result] = []          # 收集将要返回的结果
        for i in order:                     # 按分数从高到低遍历
            score = float(scores[i])        # 取当前分数（转成 Python 原生 float）
            if score < min_score:           # 分数低于阈值
                break                       # 因为已排序，后面只会更低，直接结束循环
            results.append(Result(text=self.texts[i], source=self.sources[i], score=score))  # 组装一条结果
            if len(results) >= top_k:       # 已经凑够 top_k 个
                break                       # 结束循环
        return results                      # 返回结果列表

    def __len__(self) -> int:
        # 让 len(store) 返回库里文本块的数量。
        return len(self.texts)
