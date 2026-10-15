"""全局配置。

把所有"可以调"的参数集中放在这里，初学者改一个数字就能看到效果变化。
"""

# 开启"延迟求值类型注解"：注解里的类型即使运行时尚未导入也不会报错，
# 它会被当成字符串先存起来。属于 Python 的惯例写法。
from __future__ import annotations

# dataclass：一个装饰器，能根据类里声明的字段自动生成 __init__ 等方法，
# 省去手写 self.xxx = xxx 的样板代码。
from dataclasses import dataclass

# Path：面向对象的路径处理工具，比字符串拼接更安全（自动处理跨平台分隔符）。
from pathlib import Path

# 计算项目根目录：
#   __file__      -> 当前文件 config.py 的路径
#   .resolve()    -> 转成绝对路径（消除 .. 等相对写法）
#   .parent       -> 取上一级目录；连取两次 -> 从 <项目>/simple_rag/ 到 <项目>/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 知识库目录：项目根目录下的 data 文件夹。
# 注意 `/` 运算符被 Path 重载为"拼接路径"，等价于 os.path.join。
DATA_DIR = PROJECT_ROOT / "data"


# @dataclass 会自动为下面这个类生成构造函数，例如 Settings(chunk_size=100)。
@dataclass
class Settings:
    # ---------- 文档加载 / 切块 ----------

    # 从哪里读取文档；类型是 Path，默认用上面算好的 DATA_DIR。
    data_dir: Path = DATA_DIR
    # 每个文本块的目标字符数（中文一个字算 1）。默认 220。
    chunk_size: int = 220
    # 相邻两个块之间保留的重叠字符数，避免把一句完整的话从中间切断。默认 50。
    chunk_overlap: int = 50

    # ---------- 向量化 ----------
    # "tfidf"                -> 从零手写的 TF-IDF，无需下载模型（默认）
    # "sentence-transformer" -> 语义向量，效果更好，但需要 pip 安装并下载模型

    # 向量化后端的名字，默认用 tfidf。
    embedder: str = "tfidf"
    # 当 embedder 选 sentence-transformer 时要加载的模型名（去 HuggingFace 下载）。
    st_model_name: str = "shibing624/text2vec-base-chinese"

    # ---------- 检索 ----------

    # 每次检索返回最相关的前 K 个块，默认 3 个。
    top_k: int = 3
    # 相似度低于该阈值的结果直接丢弃。0.2 是针对示例语料调出来的经验值，
    # 用来过滤"只蹭到一两个常见字词"的弱匹配；换成自己的资料后可按效果再调。
    min_score: float = 0.2

    # ---------- 生成 ----------
    # "extractive" -> 离线抽取式：把检索结果整理成带引用的答案（无需大模型）
    # "ollama"     -> 调用本地 Ollama 大模型，做真正的自然语言生成

    # 生成后端的名字，默认用离线的 extractive。
    generator: str = "extractive"
    # 使用 ollama 后端时要调用的模型名。
    ollama_model: str = "qwen2.5:7b"
    # 本地 Ollama 服务的 HTTP 接口地址。
    ollama_url: str = "http://localhost:11434/api/generate"


# 创建一个默认配置实例；其他模块 `from simple_rag.config import settings` 即可拿到它。
settings = Settings()

