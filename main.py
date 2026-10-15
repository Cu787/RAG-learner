"""命令行入口。

用法示例：

    python main.py                              # 交互式问答
    python main.py -q "公司年假有几天？"          # 提问一次后退出
    python main.py --show-chunks                # 只看切块结果，检查切得好不好
    python main.py --top-k 5                     # 检索返回 5 个片段
    python main.py --generator ollama            # 用本地 Ollama 大模型生成
    python main.py --embedder sentence-transformer
"""

# 延迟求值类型注解（说明见 config.py）。
from __future__ import annotations

# argparse：标准库里的命令行参数解析工具。
import argparse
# sys：可以修改模块搜索路径（下面会用到）、以及退出程序。
import sys
# Path：面向对象的路径处理工具。
from pathlib import Path

# 让 `python main.py` 在任意目录下运行时，都能 import 到 simple_rag 包。
# sys.path 是模块搜索路径列表；insert(0, ...) 把"本文件所在目录"插到最前面。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simple_rag.config import Settings  # noqa: E402  配置类（E402：导入需在 sys.path 设置之后）
from simple_rag.pipeline import RAGPipeline  # noqa: E402  主流程类


def parse_args() -> argparse.Namespace:
    # 创建一个参数解析器，description 会显示在 --help 里。
    parser = argparse.ArgumentParser(description="极简本地 RAG 示例项目（无需 API Key）")
    # -q/--question：传入就只问这一个问题然后退出。
    parser.add_argument("-q", "--question", help="直接提问一次，然后退出")
    # --top-k：覆盖"检索返回几个片段"，值会被转成整数。
    parser.add_argument("--top-k", type=int, help="检索返回的片段数量")
    parser.add_argument(
        "--embedder",
        choices=["tfidf", "sentence-transformer"],   # 只允许这两个取值
        help="向量化后端（默认 tfidf）",
    )
    parser.add_argument(
        "--generator",
        choices=["extractive", "ollama"],            # 只允许这两个取值
        help="生成后端（默认 extractive）",
    )
    # --show-chunks：出现这个开关（不带值）就为 True，用于查看切块结果。
    parser.add_argument("--show-chunks", action="store_true", help="打印切块结果后退出")
    # 真正解析命令行，把结果对象返回。
    return parser.parse_args()


def print_answer(pipeline: RAGPipeline, question: str) -> None:
    # 调 pipeline 拿到答案。
    result = pipeline.answer(question)
    # 打印一条分隔线（64 个等号）。
    print("\n" + "=" * 64)
    # 打印答案正文。
    print(result.text)
    # 再打印一条分隔线。
    print("=" * 64)
    # 打印"命中片段"小标题。
    print("命中片段：")
    # enumerate(..., 1) 让编号从 1 开始，方便人看。
    for i, ctx in enumerate(result.contexts, 1):
        # 把片段里的换行换成空格，并只取前 60 个字作为预览。
        preview = ctx.text.replace("\n", " ")[:60]
        # 打印：序号、来源文件、相似度（保留 3 位小数）、预览。
        print(f"  [{i}] {ctx.source}  score={ctx.score:.3f}  {preview}...")


def main() -> None:
    # 解析命令行参数。
    args = parse_args()

    # 先建一份默认配置，再按命令行选项覆盖其中若干项。
    settings = Settings()
    if args.top_k:                                # 用户传了 --top-k
        settings.top_k = args.top_k               # 覆盖 top_k
    if args.embedder:                             # 用户传了 --embedder
        settings.embedder = args.embedder         # 覆盖 embedder
    if args.generator:                            # 用户传了 --generator
        settings.generator = args.generator       # 覆盖 generator

    # 给用户一个"正在干活"的提示。
    print("⏳ 正在读取文档并构建索引 ...")
    # 用上面的配置创建并构建 RAG 流水线（build_index 在内部完成加载/切块/向量化）。
    pipeline = RAGPipeline(settings).build_index()
    # 打印索引完成的信息（describe 里含后端名、维度、块数等）。
    print(f"✅ 索引完成 —— {pipeline.describe()}")

    if args.show_chunks:                          # 只查看切块结果的分支
        print("\n下面是切块结果：")
        for i, text in enumerate(pipeline.store.texts, 1):   # 遍历每个块（编号从 1 起）
            print(f"\n[块 {i}｜来源 {pipeline.store.sources[i - 1]}]")   # 打印块号与来源
            print(text)                            # 打印块内容
        return                                     # 打印完直接结束

    if args.question:                              # 传了 -q 的分支：只问一个问题
        print_answer(pipeline, args.question)      # 打印该问题的答案
        return                                     # 结束

    # 否则进入交互模式
    print("\n进入交互问答模式：输入问题后回车；输入 exit / quit 退出。")
    while True:                                    # 无限循环，直到用户退出
        try:
            question = input("\n❓ 你的问题：").strip()   # 读取一行输入并去掉首尾空白
        except (EOFError, KeyboardInterrupt):      # 遇到 Ctrl+Z / Ctrl+C 等中断
            print("\n再见！")                       # 友好地道别
            break                                  # 跳出循环
        if not question:                           # 空输入
            continue                               # 忽略，重新提示
        if question.lower() in {"exit", "quit", "q"}:   # 输入了退出指令（忽略大小写）
            print("再见！")                          # 道别
            break                                  # 跳出循环
        print_answer(pipeline, question)           # 否则正常回答问题


# 只有当本文件被"直接运行"（python main.py）时才执行 main()；
# 被当作模块 import 时不会执行。
if __name__ == "__main__":
    main()
