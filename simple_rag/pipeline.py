"""把整个 RAG 流程串起来。

    加载文档 -> 切块 -> 向量化 -> 建索引 -> (提问时) 检索 -> 生成答案
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# dataclass：自动生成数据类的构造函数。
from dataclasses import dataclass

# 下面这些是从本项目其他模块导入的函数/类，用来拼装整条流水线：
from .chunking import split_text                                          # 切块函数
from .config import Settings, settings as default_settings                # 配置类 + 默认配置（起别名）
from .embeddings import (                                                 # 向量化相关
    TfidfEmbedder,                       # TF-IDF 向量化器（仅用于类型标注）
    SentenceTransformerEmbedder,         # 语义向量化器（仅用于类型标注）
    build_embedder,                      # 按名字创建向量化器的工厂函数
)
from .generator import build_generator   # 按名字创建生成器的工厂函数
from .loaders import load_documents      # 读取文档的函数
from .retriever import Retriever         # 检索器类
from .vector_store import Result, VectorStore   # 检索结果数据类 + 向量库类


# 自动生成 Answer(question=..., text=..., contexts=...) 构造函数。
@dataclass
class Answer:
    """一次问答的结果。"""

    question: str            # 用户问的问题
    text: str                # 生成的答案文本
    contexts: list[Result]   # 被用来回答问题的原始片段（方便调试检索效果）


class RAGPipeline:
    """RAG 主流程。"""

    def __init__(self, settings: Settings | None = None) -> None:
        # 允许外部传入配置；不传就用默认配置。
        self.settings = settings or default_settings
        # 向量化器先留空，等 build_index() 时再按配置创建。
        self.embedder: TfidfEmbedder | SentenceTransformerEmbedder | None = None
        # 现在就创建一个空的向量库对象。
        self.store = VectorStore()
        # 检索器先留空，等建库后再创建。
        self.retriever: Retriever | None = None
        # 生成器先留空，等建库后再按配置创建。
        self.generator = None
        # 记录一共切出了多少个文本块，初始为 0。
        self.num_chunks = 0

    # ------------------------------------------------------------------ #
    # 建索引：只需要做一次
    # ------------------------------------------------------------------ #
    def build_index(self) -> "RAGPipeline":
        s = self.settings   # 起个短名字，下面写起来方便

        # 1) 加载文档
        documents = load_documents(s.data_dir)   # 从 data 目录读出所有 Document
        if not documents:                        # 一篇都没读到
            raise RuntimeError(f"在 {s.data_dir} 下没有找到任何文档（.txt/.md）。")

        # 2) 切块
        chunk_texts: list[str] = []      # 存放所有小块文本
        chunk_sources: list[str] = []    # 存放每个小块对应的来源文件名（与上一行一一对应）
        for doc in documents:            # 逐篇文档
            for chunk in split_text(doc.text, s.chunk_size, s.chunk_overlap):   # 把这篇切块
                chunk_texts.append(chunk)          # 记录这一块文本
                chunk_sources.append(doc.source)   # 记录它来自哪个文件

        # 3) 向量化（先 fit 学习词表，再 encode 得到向量）
        self.embedder = build_embedder(s.embedder, s.st_model_name)   # 按配置创建向量化器
        self.embedder.fit(chunk_texts)                                # 用全部小块统计词表与 IDF
        vectors = self.embedder.encode_many(chunk_texts)              # 把所有小块批量转成向量

        # 4) 建索引
        self.store.add(vectors, chunk_texts, chunk_sources)           # 向量+文本+来源一起入库
        self.retriever = Retriever(self.embedder, self.store)         # 用同一个向量化器创建检索器
        self.generator = build_generator(s.generator, s.ollama_model, s.ollama_url)  # 按配置创建生成器
        self.num_chunks = len(chunk_texts)                            # 记下块数，方便展示
        return self                                                   # 返回自身，方便链式调用

    # ------------------------------------------------------------------ #
    # 提问：检索 + 生成
    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> Answer:
        if self.retriever is None or self.generator is None:          # 还没建索引
            raise RuntimeError("请先调用 build_index() 建立索引。")

        s = self.settings
        # 检索：拿到与问题最相关的 top_k 个片段。
        contexts = self.retriever.retrieve(question, top_k=s.top_k, min_score=s.min_score)
        # 生成：把问题和这些片段交给生成器，产出答案文本。
        text = self.generator.generate(question, contexts)
        # 把问题、答案、命中片段一起打包返回。
        return Answer(question=question, text=text, contexts=contexts)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """返回一句人话描述，方便调试。"""
        # 取向量维度；若向量化器还没建好就用 0。
        dim = self.embedder.dim() if self.embedder else 0
        # 把关键信息拼成一行字符串返回。
        return (
            f"向量后端={self.settings.embedder}，向量维度={dim}，"
            f"文本块数量={self.num_chunks}，生成后端={self.settings.generator}"
        )
