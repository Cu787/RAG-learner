"""文档加载：把 data/ 目录下的文本文件读成 Document 对象。

这一步就是 RAG 的"知识来源"——你的私有资料长什么样，先把它读进来。
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# 用来把类变成"数据类"，自动生成 __init__ 等。
from dataclasses import dataclass
# 面向对象的路径处理工具。
from pathlib import Path

# 目前只支持纯文本类文件；PDF / Word 需要额外的库（见 README 进阶部分）。
# 用集合（set）存后缀，判断"在不在里面"时速度快（平均 O(1)）。
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


# 自动生成 Document(text=..., source=...) 这样的构造函数。
@dataclass
class Document:
    """一篇原始文档。"""

    # 正文内容。
    text: str
    # 出处（这里用文件名），检索后可告诉用户"答案来自哪篇文档"。
    source: str


def load_documents(data_dir: Path) -> list[Document]:
    """递归读取 data_dir 下所有受支持的文本文件。"""

    # 把入参统一转成 Path，避免调用方传字符串时下面的 .exists() 报错。
    data_dir = Path(data_dir)

    # 目录不存在就立刻抛异常，并给出明确提示（fail fast，早发现早解决）。
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    # 准备一个空列表，用来装入读到的所有文档。
    documents: list[Document] = []

    # rglob("*") 递归遍历目录下所有条目；sorted() 保证顺序稳定、结果可复现。
    for path in sorted(data_dir.rglob("*")):
        # 只处理"是文件"且"后缀在支持列表里"的条目（跳过子目录、跳过不支持的类型）。
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            # 以 UTF-8 读取整个文件，并去掉首尾空白（换行/空格）。
            text = path.read_text(encoding="utf-8").strip()
            # 去掉空白后若还有内容，才认为它是一篇有效文档。
            if text:
                # 把正文和文件名一起打包成 Document，追加进列表。
                documents.append(Document(text=text, source=path.name))

    # 返回收集好的文档列表。
    return documents

