"""向量化（embedding）：把文本变成一串数字（向量），这样计算机才能"计算相似度"。

本项目提供两种后端，接口完全一致，可以随时替换：

  1) TfidfEmbedder —— 从零手写 TF-IDF，只依赖 numpy，不下载任何模型。
     * 优点：秒级运行、完全离线、原理透明，非常适合入门。
     * 缺点：只匹配"词"，不懂"语义"（"苹果"和"iPhone"它认为是两个无关的词）。

  2) SentenceTransformerEmbedder —— 真正的语义向量模型（可选，需额外安装）。
     * 会把语义相近的句子映射到相近的向量，检索质量更好。

接口约定：
  embedder.fit(corpus)       学习词表 / 加载模型
  embedder.encode(text)      单条文本 -> 向量 (numpy.ndarray)
  embedder.encode_many(list) 多条文本 -> 向量矩阵
  embedder.dim()             向量维度
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# math：做对数运算（计算 IDF 时要用 log）。
import math
# re：正则表达式，用来切出英文单词和中文单字。
import re
# defaultdict(int)：一个"取不存在的键时默认返回 0"的字典，统计词频很方便。
from collections import defaultdict

# numpy：向量 / 矩阵运算的核心库。
import numpy as np

# 匹配英文单词（连续字母）或数字（连续数字）的正则。
_WORD_RE = re.compile(r"[a-zA-Z]+|\d+")
# 匹配任意一个中文字符的正则（\u4e00-\u9fff 是常用汉字区间）。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """极简分词：英文按单词，中文按 单字 + 相邻双字(bigram)。

    中文词之间没有空格，如果不做处理，"年假" 和 "年假天数" 就很难对上。
    这里用"相邻两个汉字"组成一个虚拟词，能在不安装 jieba 的情况下
    大幅提升中文匹配效果。这是一个工程上的折中，够用且零依赖。
    """

    text = (text or "").lower()                  # 防止 None；并统一转小写（英文不区分大小写）
    tokens: list[str] = _WORD_RE.findall(text)   # 先用正则捞出所有英文单词 / 数字

    cjk = _CJK_RE.findall(text)                 # 捞出所有中文字符（逐个）
    tokens.extend(cjk)                                             # 每个中文字也作为一个 token（单字）
    tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))  # 相邻两字拼成 bigram（双字）
    return tokens                               # 返回合并后的 token 列表


class TfidfEmbedder:
    """从零实现的 TF-IDF 向量化器。

    TF  (Term Frequency)   : 某个词在这段文本里出现得多不多
    IDF (Inverse Doc Freq) : 某个词在整个语料里稀不稀有（越稀有越重要）
    两者相乘 -> 权重；再做 L2 归一化 -> 向量长度为 1，之后点积即余弦相似度。
    """

    name = "tfidf"   # 后端标识名，供 build_embedder 识别

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}          # 词 -> 向量下标 的映射表（fit 时填充）
        self.idf: np.ndarray | None = None       # 每个词的 IDF 权重数组（fit 时填充）

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        """统计语料，构建词表与 IDF。"""
        doc_freq: dict[str, int] = defaultdict(int)   # 记录"某个词出现在了多少篇文档里"
        for doc in corpus:                            # 遍历每一篇文档
            for token in set(tokenize(doc)):          # set() -> 同一篇里重复的词只算一次
                doc_freq[token] += 1                  # 该词的文档计数 +1

        # 词表：把所有出现过的词按字典序排序后编号，保证多次运行结果稳定。
        self.vocab = {token: i for i, token in enumerate(sorted(doc_freq))}

        n_docs = max(1, len(corpus))                        # 文档总数（max 防止除以 0）
        idf = np.zeros(len(self.vocab), dtype=np.float32)   # 先建一个全 0 的 IDF 数组
        for token, idx in self.vocab.items():               # 逐个词计算 IDF
            # 平滑 IDF 公式：log((1+N)/(1+df)) + 1，避免除零，且保证结果恒为正。
            idf[idx] = math.log((1 + n_docs) / (1 + doc_freq[token])) + 1.0
        self.idf = idf                                 # 保存 IDF 数组
        return self                                    # 返回自身，方便链式调用 .fit().encode()

    def _term_frequency(self, text: str) -> np.ndarray:
        # 先建一个与词表等长的全 0 向量。
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        tokens = tokenize(text)               # 把这句文本分词
        if not tokens:                        # 一个 token 都没有
            return vec                        # 直接返回全 0 向量
        for token in tokens:                  # 遍历每个 token
            idx = self.vocab.get(token)       # 查它在词表里的下标（不在词表里会得到 None）
            if idx is not None:               # 只有词表里有的词才统计
                vec[idx] += 1.0               # 对应位置计数 +1（这就是词频 TF）
        vec /= len(tokens)                    # 除以 token 总数做长度归一化（避免长文本天然占优）
        return vec                            # 返回词频向量

    def encode(self, text: str) -> np.ndarray:
        if self.idf is None:                  # 还没 fit 就直接 encode
            raise RuntimeError("请先调用 fit() 构建词表，再调用 encode()。")
        vec = self._term_frequency(text) * self.idf   # TF × IDF = TF-IDF 权重向量
        norm = float(np.linalg.norm(vec))             # 求向量模长
        if norm > 0:                                  # 只对非零向量做归一化
            vec = vec / norm                          # L2 归一化：让向量长度为 1
        return vec                                    # 返回向量（归一化后点积即余弦相似度）

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:                                               # 空输入
            return np.zeros((0, len(self.vocab)), dtype=np.float32)  # 返回 0 行 D 列的空矩阵
        return np.vstack([self.encode(t) for t in texts])           # 逐条编码后竖直堆成矩阵

    def dim(self) -> int:
        # 向量维度 = 词表大小。
        return len(self.vocab)


class SentenceTransformerEmbedder:
    """可选的语义向量后端（需要 `pip install sentence-transformers`）。"""

    name = "sentence-transformer"   # 后端标识名

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese") -> None:
        # 延迟导入：只有真正用到时才需要这个依赖
        try:
            from sentence_transformers import SentenceTransformer   # 尝试导入第三方库
        except ImportError as exc:  # 给初学者一个友好的提示，而不是一大段 traceback
            raise ImportError(
                "尚未安装 sentence-transformers。请先执行：\n"
                "    pip install sentence-transformers\n"
                "然后重新运行（首次运行会自动下载模型，需要联网）。\n"
                "如果暂时不想安装，用默认的 --embedder tfidf 即可。"
            ) from exc   # 用 from exc 保留原始异常链，方便排查根因

        self.model = SentenceTransformer(model_name)   # 加载（或从本地缓存读取）预训练模型

    def fit(self, corpus: list[str]) -> "SentenceTransformerEmbedder":
        # 语义模型是预训练好的，不需要在语料上"拟合"。
        return self   # 直接返回自身，与 TfidfEmbedder 接口保持一致

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(                                                   # 把结果转成 numpy 数组
            self.model.encode(text, normalize_embeddings=True), dtype=np.float32
        )   # normalize_embeddings=True 让输出向量长度为 1，便于后续用点积算余弦相似度

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return np.asarray(                                                   # 批量编码，一次性编码多条更高效
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    def dim(self) -> int:
        # 取该模型输出向量的维度。
        return int(self.model.get_sentence_embedding_dimension())


def build_embedder(name: str, model_name: str = "") -> TfidfEmbedder | SentenceTransformerEmbedder:
    """根据配置名创建向量化器。"""
    name = (name or "tfidf").lower()    # 兜底为空串时用 tfidf；并忽略大小写
    if name == "tfidf":                 # 选择了 TF-IDF 后端
        return TfidfEmbedder()          # 直接构造一个 TF-IDF 向量化器
    if name in ("sentence-transformer", "st", "sentence_transformer"):   # 几个等价的写法
        return SentenceTransformerEmbedder(model_name or "shibing624/text2vec-base-chinese")
    raise ValueError(f"未知的 embedder：{name}（可选：tfidf / sentence-transformer）")   # 其它名字报错
