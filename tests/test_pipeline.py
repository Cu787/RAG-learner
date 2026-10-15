"""基础单元测试。

运行方式（在项目根目录下）：
    python -m unittest discover -s tests -v
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# sys：用来修改模块搜索路径，保证测试能 import 到 simple_rag 包。
import sys
# unittest：Python 自带的单元测试框架。
import unittest
# Path：路径处理工具。
from pathlib import Path

# 把项目根目录加入 import 路径：parent 是 tests/，再 parent 就是项目根目录。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_rag.chunking import split_text  # noqa: E402  被测试的切块函数
from simple_rag.embeddings import TfidfEmbedder, tokenize  # noqa: E402  向量化器与分词函数
from simple_rag.pipeline import RAGPipeline  # noqa: E402  被测试的主流程
from simple_rag.vector_store import VectorStore  # noqa: E402  被测试的向量库


class TestChunking(unittest.TestCase):   # 测试分组一：切块
    def test_split_produces_non_empty_chunks(self):
        # 造一段较长的文本（同一句话重复 20 遍）。
        text = "这是第一句话。这是第二句话。这是第三句话。" * 20
        # 用较小的 chunk_size 去切。
        chunks = split_text(text, chunk_size=50, chunk_overlap=10)
        self.assertTrue(chunks)                                   # 断言：切出了内容（列表非空）
        self.assertTrue(all(chunk.strip() for chunk in chunks))   # 断言：没有任何一块是纯空白

    def test_empty_text(self):
        # 空字符串应当得到空列表。
        self.assertEqual(split_text(""), [])


class TestEmbeddings(unittest.TestCase):   # 测试分组二：向量化
    def test_tokenize_mixes_chinese_and_english(self):
        tokens = tokenize("年假 15 days")   # 中英数字混合的输入
        self.assertIn("年", tokens)      # 中文单字
        self.assertIn("年假", tokens)    # 中文双字
        self.assertIn("days", tokens)    # 英文单词
        self.assertIn("15", tokens)      # 数字

    def test_most_similar_document_ranks_first(self):
        corpus = [                       # 三篇主题各异的"文档"
            "公司年假一共有十五天，可以分段使用。",
            "产品支持 Windows 和 macOS 以及 Linux。",
            "报销需要提供发票，并在三十天内提交。",
        ]
        embedder = TfidfEmbedder().fit(corpus)   # 在语料上训练 TF-IDF
        store = VectorStore()                    # 新建一个空向量库
        store.add(embedder.encode_many(corpus), corpus, ["a", "b", "c"])   # 入库（来源用 a/b/c 标记）

        results = store.search(embedder.encode("年假有几天"), top_k=1)   # 检索"年假"相关问题
        self.assertEqual(len(results), 1)              # 断言：只返回 1 条
        self.assertEqual(results[0].source, "a")       # 断言：命中的是第 1 篇（来源 a）
        self.assertGreater(results[0].score, 0.0)      # 断言：相似度大于 0


class TestPipeline(unittest.TestCase):   # 测试分组三：端到端流程
    @classmethod
    def setUpClass(cls):
        # 整个测试类开始前只建一次索引（较耗时，且结果可复用）。
        cls.pipeline = RAGPipeline().build_index()

    def test_index_built(self):
        # 断言：索引里确实切出了文本块。
        self.assertGreater(self.pipeline.num_chunks, 0)

    def test_answer_returns_contexts(self):
        answer = self.pipeline.answer("公司年假有几天？")   # 问一个问题
        self.assertTrue(answer.contexts)   # 断言：既有命中片段
        self.assertTrue(answer.text)       # 断言：又生成了答案文本


# 直接运行本文件时，用详细模式（verbosity=2）跑测试。
if __name__ == "__main__":
    unittest.main(verbosity=2)
