"""文本切块（chunking）。

为什么不整篇文档直接检索？
  1. 大模型/相似度对"太长"的文本不敏感，一小段话反而更精准；
  2. 检索粒度细，命中的片段更贴合问题；
  3. 拼进提示词时也更省 token。

策略：
  * 以"行"为最小单位（Markdown 的标题、列表项都能保持完整）；
  * 把若干行累加到接近 chunk_size 就切成一块，行与行之间用换行拼接；
  * 单个超长的行再按句号二次切分；
  * 相邻块之间保留 chunk_overlap 个字符的重叠，避免语义被生硬切断。
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# 正则表达式模块，用来按标点切分句子。
import re

# 预编译一个正则：匹配"句末标点之后"的位置。
# (?<=...) 是"后行断言"(lookbehind)，表示"该位置前面是这些字符之一"，
# 它本身不消耗字符，只标记一个切分点。
_SENTENCE_SEP = re.compile(r"(?<=[。！？!?；;])")


def _split_units(text: str) -> list[str]:
    """把文本拆成"行"级别的单元（去空行）。"""
    # text.split("\n") 先按换行切成多行；
    # line.strip() 去掉每行首尾空白；
    # if line.strip() 过滤掉空行；
    # 整个是列表推导式，一次性生成结果列表。
    return [line.strip() for line in text.split("\n") if line.strip()]


def _force_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """把一行超长文本按句子切，句子仍超长就硬切。"""
    pieces: list[str] = []                  # 收集切好的小片段
    current = ""                            # 当前正在累积的片段
    for sent in _SENTENCE_SEP.split(text):  # 按句末标点把整行切成一句一句
        sent = sent.strip()                 # 去掉句子首尾空白
        if not sent:                        # 空句子（连续标点可能产生）跳过
            continue                        # 进入下一轮循环
        if len(sent) > chunk_size:          # 单个句子仍然比目标长度还长
            if current:                     # 若手上还有没存下的片段
                pieces.append(current)      # 先把它存进结果
                current = ""                # 再清空
            step = max(1, chunk_size - chunk_overlap)   # 每次前进的步长，用 max 防止变成 0 或负数
            for i in range(0, len(sent), step):         # 从头开始，按步长滑动窗口
                pieces.append(sent[i : i + chunk_size])  # 取长度 chunk_size 的一截
            continue                        # 这个长句处理完毕，进入下一句
        if not current:                     # 手上是空的时候
            current = sent                  # 直接以这句作为开头
        elif len(current) + len(sent) <= chunk_size:   # 拼上去不会超长
            current += sent                 # 就拼上（中文句子可以无缝相接）
        else:                               # 拼上去会超长
            pieces.append(current)          # 先把当前片段存起来
            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""   # 取结尾的重叠部分
            current = overlap + sent        # 新片段 = 重叠内容 + 这句
    if current.strip():                     # 循环结束后若还有剩余内容
        pieces.append(current)              # 补进结果
    return pieces                           # 返回所有片段


def split_text(text: str, chunk_size: int = 220, chunk_overlap: int = 50) -> list[str]:
    """把一段长文本切成若干块。"""

    text = (text or "").strip()    # 防止传入 None；并去掉首尾空白
    if not text:                   # 文本为空
        return []                  # 直接返回空列表

    chunks: list[str] = []         # 最终结果列表
    current = ""                   # 正在累积的当前块

    for unit in _split_units(text):    # 逐"行"处理
        # 超长的一行：先给当前块收尾，再单独把这行切细
        if len(unit) > chunk_size:
            if current:                # 当前块有内容
                chunks.append(current)   # 先收尾保存
                current = ""             # 并清空
            chunks.extend(_force_split(unit, chunk_size, chunk_overlap))   # 把这行切细后追加
            continue                   # 进入下一行

        if not current:                # 当前块还是空的
            current = unit             # 直接用这一行开头
        elif len(current) + 1 + len(unit) <= chunk_size:   # +1 是行间要加的那个换行符
            current = current + "\n" + unit                # 用换行拼接，保留 Markdown 结构
        else:                          # 再加一行就超长
            chunks.append(current)     # 先把当前块存起来
            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""   # 结尾的重叠部分
            current = (overlap + "\n" + unit) if overlap else unit            # 新块 = 重叠 + 这一行

    if current.strip():                # 收尾最后一块
        chunks.append(current)

    # 去掉每块首尾空白、并过滤掉空块后返回。
    return [c.strip() for c in chunks if c.strip()]
