"""simple_rag —— 一个从零实现的极简 RAG（检索增强生成）教学项目。

模块划分（对应 RAG 的流水线）：

    loaders.py       文档加载        data/*.md  ->  Document
    chunking.py      文本切块        Document    ->  chunk 列表
    embeddings.py    向量化          chunk       ->  向量
    vector_store.py  向量库 + 检索   向量        ->  Top-K 相似片段
    retriever.py     检索器封装      问题        ->  相关片段
    generator.py     生成器          片段        ->  自然语言答案
    pipeline.py      串起全流程
"""

# 从同级的 config 模块导入两个名字：Settings 类、settings 全局实例。
# 这样外部既可用 `from simple_rag import Settings` 自定义配置，
# 也可用 `from simple_rag import settings` 直接拿默认配置。
from .config import Settings, settings

# __all__ 声明"别人写 from simple_rag import * 时会导出哪些名字"。
# 它不影响运行，只是对外公开接口的一张清单。
__all__ = ["Settings", "settings"]

# 版本号字符串，方便日后排查"我用的是哪个版本"。
__version__ = "0.1.0"

