"""生成器：把"检索到的片段"变成最终的答案。

一个真正的 RAG 会把 片段 + 问题 拼成提示词，交给大模型生成答案。
本文件给了两种实现：

  ExtractiveGenerator  —— 默认。不依赖大模型，直接把检索到的片段
                          整理成"带出处的资料清单"。虽然不"通顺"，
                          但它让你先看清"检索"这一半是否工作正常。

  OllamaGenerator      —— 可选。调用本地 Ollama（免费、离线）跑真正的
                          LLM 生成。需要先安装 Ollama 并 `ollama pull qwen2.5:7b`。
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# json：虽然这里直接用 requests 的 .json()，保留导入便于扩展/调试。
import json

# 从向量库模块导入检索结果数据类，用于类型标注。
from .vector_store import Result

# 喂给大模型的提示词模板。注意 {context} 里是检索到的参考资料。
# 三个要点：① 只根据资料回答 ② 没答案就说没有 ③ 标注资料编号。
PROMPT_TEMPLATE = """你是一个严谨的问答助手。请【仅根据】下面的参考资料回答用户问题。
如果参考资料中没有答案，就明确回答"根据现有资料无法回答"，不要编造。

参考资料：
{context}

用户问题：{question}

请用简洁的中文回答，并在句末标注用到的资料编号，例如 [1]、[2]。"""


def build_context(contexts: list[Result]) -> str:
    """把检索结果拼成提示词里的"参考资料"文本。"""
    blocks = []                                    # 用来装每条资料的字符串
    for i, ctx in enumerate(contexts, 1):          # 编号从 1 开始，方便模型引用
        blocks.append(f"[{i}] (来源：{ctx.source}) {ctx.text}")   # 拼成 "[1] (来源：xxx) 正文"
    return "\n\n".join(blocks)                     # 用空行把各条资料连接起来


class ExtractiveGenerator:
    """离线抽取式生成器：不需要大模型。"""

    name = "extractive"   # 后端标识名

    def generate(self, question: str, contexts: list[Result]) -> str:
        if not contexts:                              # 一个片段都没检索到
            return "抱歉，知识库里没有找到与该问题相关的内容。"

        # 第一行先说明"关于什么问题、找到几条资料"。
        lines = [f"关于「{question}」，在知识库中找到 {len(contexts)} 条相关资料：\n"]
        for i, ctx in enumerate(contexts, 1):         # 逐条整理，编号从 1 开始
            lines.append(f"[{i}]（来源：{ctx.source}，相似度 {ctx.score:.3f}）")   # 加一行标题（含来源与分数）
            lines.append(f"    {ctx.text}\n")          # 再贴出片段正文（缩进 4 格）
        return "\n".join(lines).rstrip()              # 拼成完整文本并去掉末尾多余空白


class OllamaGenerator:
    """调用本地 Ollama 大模型做真正的生成。"""

    name = "ollama"   # 后端标识名

    def __init__(self, model: str = "qwen2.5:7b", url: str = "http://localhost:11434/api/generate") -> None:
        self.model = model   # 记住要用的模型名
        self.url = url       # 记住 Ollama 的接口地址

    def generate(self, question: str, contexts: list[Result]) -> str:
        import requests   # requests 是可选依赖，用到时才导入

        if not contexts:                              # 没有检索到任何内容
            return "抱歉，知识库里没有找到与该问题相关的内容。"

        # 用模板把"参考资料 + 问题"拼成最终提示词。
        prompt = PROMPT_TEMPLATE.format(
            context=build_context(contexts), question=question
        )

        try:
            response = requests.post(                       # 向 Ollama 发一个 POST 请求
                self.url,
                json={"model": self.model, "prompt": prompt, "stream": False},   # 请求体：模型、提示词、非流式
                timeout=120,                                # 最多等 120 秒
            )
            response.raise_for_status()                     # 如果 HTTP 状态码是错误码，抛出异常
            # 从返回的 JSON 里取 "response" 字段作为答案；取不到就留空。
            return response.json().get("response", "").strip() or "(模型没有返回内容)"
        except Exception as exc:  # noqa: BLE001 - 教学项目，统一给出友好提示
            # 任何失败（没装 Ollama、没启动、模型没拉取等）都返回可读的说明。
            return (
                "调用 Ollama 失败："
                f"{exc}\n"
                "请确认已安装并启动 Ollama（https://ollama.com），"
                f"并执行过 `ollama pull {self.model}`；"
                "或改用 --generator extractive 离线模式。"
            )


def build_generator(name: str, model: str = "", url: str = ""):
    """根据配置名创建生成器。"""
    name = (name or "extractive").lower()   # 兜底为空串时用 extractive；并忽略大小写
    if name == "extractive":                # 选择了离线抽取式
        return ExtractiveGenerator()        # 直接构造
    if name == "ollama":                    # 选择了 Ollama
        return OllamaGenerator(model=model or "qwen2.5:7b", url=url or "http://localhost:11434/api/generate")
    raise ValueError(f"未知的 generator：{name}（可选：extractive / ollama）")   # 其它名字报错
