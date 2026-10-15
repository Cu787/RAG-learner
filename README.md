# 从零开始的简易 RAG 项目

> 一个**纯本地运行、无需任何 API Key、零门槛**的检索增强生成（RAG）教学项目。
> 面向"从零开始的学习者"：代码短、注释多、每个模块只做一件事，跑起来就能看到效果。

---

## 一、一句话理解 RAG

**RAG = 检索（Retrieval）+ 增强（Augmented）+ 生成（Generation）**

大模型本身不知道你的私有资料（公司手册、产品文档、你的笔记……）。
RAG 的做法是：**先去你的资料库里"查"出最相关的几段，再让大模型/程序"照着小抄回答"**。

这样既能回答私有问题，又能减少"胡编乱造"（幻觉）。

## 二、整体流程

```
                    ┌─────────────── 建立索引（只做一次）───────────────┐
   你的文档们  ──▶  加载 ──▶ 切块 ──▶ 向量化 ──▶ 存进向量库
   (data/*.md)     load    chunk   embedding   vector store
                    └──────────────────────────────────────────────┘
                                                │
   用户提问  ──▶  向量化  ──▶  相似度检索 Top-K  ◀─┘
                                   │
                                   ▼
                     把「问题 + 命中的资料」拼成提示词
                                   │
                                   ▼
                          大模型 / 程序  ──▶  最终答案
```

对应到本项目的模块：

| 步骤 | 文件 | 作用 |
| --- | --- | --- |
| 加载 | `simple_rag/loaders.py` | 读取 `data/` 下的 `.md` / `.txt` |
| 切块 | `simple_rag/chunking.py` | 长文档切成一小段一小段 |
| 向量化 | `simple_rag/embeddings.py` | 文本 → 数字向量（TF-IDF，手写） |
| 向量库 | `simple_rag/vector_store.py` | 存向量 + 余弦相似度检索（手写） |
| 检索 | `simple_rag/retriever.py` | 封装"问题 → Top-K 片段" |
| 生成 | `simple_rag/generator.py` | 片段 → 答案（离线 / Ollama） |
| 串联 | `simple_rag/pipeline.py` | 把上面全部串起来 |

## 三、快速开始

### 1. 环境要求

- Python 3.10+（本项目在 **Python 3.12.10** 上验证通过）
- 唯一必需依赖：`numpy`

### 2. 安装依赖

```powershell
cd simple-rag
pip install -r requirements.txt
```

### 3. 运行

```powershell
# 交互式问答（输入问题回车，输入 exit 退出）
python main.py

# 只问一个问题
python main.py -q "公司年假有几天？"

# 看看文档被切成了什么样（调试切块效果）
python main.py --show-chunks
```

你应当看到类似输出：

```
⏳ 正在读取文档并构建索引 ...
✅ 索引完成 —— 向量后端=tfidf，向量维度=958，文本块数量=8，生成后端=extractive

================================================================
关于「公司年假有几天？」，在知识库中找到 3 条相关资料：

[1]（来源：company_handbook.md，相似度 0.407）
    # 星野科技 员工手册（节选）
    ## 一、年假政策
    正式员工入职满 1 年可享受 5 天带薪年假；满 3 年增加到 10 天；满 5 年及以上增加到 15 天。
    ...
================================================================
```

> 试试这几个问题：`公司年假有几天？`、`云笔记支持哪些平台？`、`忘记密码怎么办？`、`出差住宿报销标准是多少？`

## 四、目录结构

```
simple-rag/
├── README.md                 ← 你正在读的文件
├── requirements.txt          ← 必需依赖（numpy）
├── requirements-optional.txt ← 可选进阶依赖
├── main.py                   ← 命令行入口
├── data/                     ← 你的知识库（放自己的资料在这里！）
│   ├── company_handbook.md
│   ├── product_guide.md
│   └── faq.txt
├── docs/
│   └── TUTORIAL.md           ← 更详细的原理讲解
├── simple_rag/               ← 核心代码包
│   ├── __init__.py
│   ├── config.py             ← 所有可调参数
│   ├── loaders.py            ← 1. 加载文档
│   ├── chunking.py           ← 2. 切块
│   ├── embeddings.py         ← 3. 向量化
│   ├── vector_store.py       ← 4. 向量库 + 检索
│   ├── retriever.py          ← 5. 检索器
│   ├── generator.py          ← 6. 生成器
│   └── pipeline.py           ← 7. 串联全流程
└── tests/
    └── test_pipeline.py      ← 单元测试
```

## 五、核心原理（对照代码看）

### 1. 切块（chunking）
一整篇文档太长，直接拿去检索不精准，所以先切成小块。
本项目以"行"为单位累加，接近 `chunk_size` 就切成一块，并让相邻块保留
`chunk_overlap` 个字符的重叠（`chunking.py`）。用 `--show-chunks` 能直观看到效果。

### 2. 向量化（embedding）
把文本变成一串数字（向量），让计算机能"算相似度"。本项目手写了 **TF-IDF**：

- **TF**：一个词在这段文本里出现的频率；
- **IDF**：一个词在整个资料库里越稀有，越有区分度（权重越高）；
- 两者相乘 → 权重向量，再做 L2 归一化（`embeddings.py`）。

> 中文没有空格，怎么分词？本项目用「**中文单字 + 相邻双字(bigram)**」的巧办法，
> 不依赖 jieba 也能有不错的效果。

### 3. 检索（vector_store.py）
用**余弦相似度**衡量两个向量的方向有多接近：

```
cos(a, b) = (a · b) / (|a| * |b|)      取值范围 [-1, 1]，越接近 1 越相似
```

对所有块算一遍相似度，取分数最高的 Top-K 个，就是"检索结果"。

### 4. 生成（generator.py）
把检索到的片段和问题拼成提示词，交给"生成器"：
- **extractive（默认）**：不依赖大模型，直接把命中片段整理成带出处的清单。
  它不"通顺"，但能让你先确认**检索这一半是否正确**——这是学习 RAG 最重要的一步。
- **ollama（可选）**：调用本地大模型，生成真正通顺的答案。

## 六、可调参数（都在 `simple_rag/config.py`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `chunk_size` | 220 | 每块的目标字符数。调大→上下文更全；调小→更精准 |
| `chunk_overlap` | 50 | 相邻块的重叠字符数，避免语义被切断 |
| `top_k` | 3 | 每次检索返回几段 |
| `min_score` | 0.2 | 相似度低于此值的片段丢弃（过滤"蹭到几个常见字"的弱匹配）。该值依赖语料，建议自行微调 |
| `embedder` | `tfidf` | 向量化后端，可换 `sentence-transformer` |
| `generator` | `extractive` | 生成后端，可换 `ollama` |

命令行也能临时覆盖，例如：`python main.py --top-k 5`

## 七、进阶玩法

### A. 换成"语义向量"（检索更聪明）
TF-IDF 只认"词"，不懂"语义"（"苹果"和"iPhone"它觉得没关系）。换成神经网络语义模型：

```powershell
pip install sentence-transformers
python main.py --embedder sentence-transformer
```

### B. 接入本地大模型做真正的生成（免费、离线）
1. 安装 [Ollama](https://ollama.com)；
2. 拉一个模型：`ollama pull qwen2.5:7b`；
3. 运行：

```powershell
python main.py --generator ollama
```

> `generator.py` 里的 `PROMPT_TEMPLATE` 就是经典的 RAG 提示词模板，值得一读。

### C. 换成专业向量库
把 `simple_rag/vector_store.py` 换成 **FAISS / Chroma / Milvus** 即可——`pipeline.py` 之外的接口不用动。

### D. 支持 PDF / Word
在 `loaders.py` 增加 `.pdf` 解析（如 `pypdf`）、`.docx` 解析（如 `python-docx`）即可。

### E. 放自己的资料
把任意 `.md` / `.txt` 文件丢进 `data/` 文件夹，重新运行即可，无需改代码。

### F. 换成 LangChain（用框架重写一遍）
想看看"用框架"是什么体验、"效果"会比手写更好吗？见后文 **《十一、用 LangChain 重写一遍（用法与效果）》**。

## 八、运行测试

```powershell
python -m unittest discover -s tests -v
```

覆盖了切块、中文分词、相似度排序、端到端问答等 6 个用例。

## 九、常见问题

**Q：提示 `ModuleNotFoundError: No module named 'numpy'`？**
A：执行 `pip install -r requirements.txt`。

**Q：检索结果不准？**
A：① 调大 `top_k`；② 检查切块是否合理（`--show-chunks`）；③ 升级到语义向量（进阶 A）。

**Q：中文效果一般？**
A：TF-IDF 是"词匹配"，本就弱于语义模型。生产环境建议用 `sentence-transformer` 或在线 embedding。

**Q：没有 API Key 也能用吗？**
A：完全可以。默认全流程离线，不需要任何 Key。

## 十、给学习者的练习

1. **改数据**：把 `data/` 换成你自己的笔记，观察检索效果。
2. **改切块**：把 `chunk_size` 改成 100 / 400，看 `--show-chunks` 和答案的变化。
3. **改打分**：把 `vector_store.py` 的余弦相似度换成"点积"或"欧氏距离"，对比排序差异。
4. **加引用**：在 `ExtractiveGenerator` 里只输出命中的**句子**而不是整块。
5. **接 Ollama**：跑通进阶 B，感受"检索质量"如何影响"生成质量"。

---

## 十一、用 LangChain 重写一遍（用法与效果）

> 前十章，我们手写了 RAG 的每一个零件。这一章换个视角：**用业界最流行的框架 LangChain 做同一件事，"用法"有什么不同、"效果"会不会更好？**
>
> 先给结论：**LangChain 改变的是"工程效率"，不是"检索质量"。** 原因见第 5 节。

### 1. 先摆正位置：LangChain 是什么

LangChain 不是"更聪明的检索"，而是一层**编排（orchestration）胶水**——把你前面手写的零件抽象成统一接口，于是"换零件"的成本大幅降低。对照关系：

| 本项目（手写） | LangChain 里的等价物 | 导入路径 |
| --- | --- | --- |
| `loaders.py` | `TextLoader` / `DirectoryLoader` / `PyPDFLoader` | `langchain_community.document_loaders` |
| `chunking.py` | `RecursiveCharacterTextSplitter` | `langchain_text_splitters` |
| `embeddings.py` 的接口约定 | `Embeddings` 抽象基类 | `langchain_core.embeddings` |
| `embeddings.py` 的语义后端 | `OpenAIEmbeddings` / `OllamaEmbeddings` / `HuggingFaceEmbeddings` | `langchain_openai` / `langchain_ollama` / `langchain_huggingface` |
| `vector_store.py` | `InMemoryVectorStore` / `FAISS` / `Chroma` | `langchain_core.vectorstores` / `langchain_community.vectorstores` |
| `retriever.py` | `VectorStoreRetriever`（由 `store.as_retriever()` 得到） | — |
| `generator.py` 的 `PROMPT_TEMPLATE` | `ChatPromptTemplate` | `langchain_core.prompts` |
| `generator.py` 的 `OllamaGenerator` | `ChatOllama` | `langchain_ollama` |
| `pipeline.py`（串联全流程） | LCEL 的 `\|` 运算符 | `langchain_core.runnables` |

### 2. 安装

LangChain 现在拆成了很多小包，**按需安装**即可：

```powershell
# 最小组合：能跑"检索 + 离线拼接"，完全不需要 API Key
pip install langchain-core langchain-text-splitters

# 接本地 Ollama 大模型
pip install langchain-ollama

# 接在线大模型（OpenAI 兼容）
pip install langchain-openai

# 换专业向量库
pip install langchain-community faiss-cpu
```

> 导入路径会随大版本变动。若报 `ModuleNotFoundError`，请以**你实际安装的版本**的官方 API 文档为准。

### 3. 对照实现：同一件事，两套写法

下面把"加载 → 切块 → 向量化 → 检索 → 生成"用 LangChain 精简重写一遍。
**关键：向量化器和提示词直接复用本项目已有的代码**，只把"编排"换成 LangChain——这样对比才公平。
把下面这段另存为项目根目录的 `langchain_demo.py`（仅演示用，不属于核心包）：

```python
"""LangChain 版 RAG 演示：复用本项目的数据、TF-IDF 与提示词，只替换"编排层"。

    python langchain_demo.py -q "公司年假有几天？"                     # 离线，无需 Key
    python langchain_demo.py -q "..." --llm ollama  --model qwen2.5:7b
    python langchain_demo.py -q "..." --llm openai  --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 同 main.py，保证能 import simple_rag

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from simple_rag.config import Settings            # 复用项目配置（chunk_size / top_k ...）
from simple_rag.embeddings import TfidfEmbedder    # 复用项目手写的 TF-IDF
from simple_rag.generator import PROMPT_TEMPLATE   # 复用项目同一个提示词
from simple_rag.loaders import load_documents      # 复用项目的加载器


class TfidfEmbeddings(Embeddings):
    """把手写的 TfidfEmbedder 适配成 LangChain 的 Embeddings 接口。"""

    def __init__(self) -> None:
        self._impl = TfidfEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._impl.fit(texts)                       # TF-IDF 必须先"学词表"
        return [vec.tolist() for vec in self._impl.encode_many(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._impl.encode(text).tolist()


def format_docs(docs: list[Document]) -> str:
    """把检索片段拼成"参考资料"（格式与项目的 build_context 一致）。"""
    return "\n\n".join(
        f"[{i}] (来源：{d.metadata.get('source', '?')}) {d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def build_chain(settings: Settings, llm_mode: str, model: str):
    # 1) 加载 + 切块（改用 LangChain 的递归字符切块器）
    raw_docs = [
        Document(page_content=d.text, metadata={"source": d.source})
        for d in load_documents(settings.data_dir)
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,          # 与项目同为 220
        chunk_overlap=settings.chunk_overlap,    # 与项目同为 50
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],   # 照顾中文标点
    )
    splits = splitter.split_documents(raw_docs)

    # 2) 向量化 + 内存向量库 + 检索器
    store = InMemoryVectorStore.from_documents(splits, embedding=TfidfEmbeddings())
    retriever = store.as_retriever(search_kwargs={"k": settings.top_k})

    # 3) 组装链：LCEL 的 `|` 就是把上一步的输出喂给下一步
    if llm_mode == "extractive":                 # 离线：等价于项目的 ExtractiveGenerator
        def extractive(question: str) -> str:
            hits = retriever.invoke(question)
            return f"关于「{question}」，命中 {len(hits)} 条资料：\n\n" + format_docs(hits)
        return retriever, RunnableLambda(extractive)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)   # 复用项目提示词
    if llm_mode == "ollama":
        from langchain_ollama import ChatOllama       # 可选依赖，用到才导入
        llm = ChatOllama(model=model or "qwen2.5:7b")
    else:
        from langchain_openai import ChatOpenAI       # 可选依赖，用到才导入
        llm = ChatOpenAI(model=model or "gpt-4o-mini")
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return retriever, chain


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain 版 RAG 演示")
    parser.add_argument("-q", "--question")
    parser.add_argument("--llm", choices=["extractive", "ollama", "openai"], default="extractive")
    parser.add_argument("--model", default="")
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    settings = Settings()
    if args.top_k:
        settings.top_k = args.top_k

    retriever, chain = build_chain(settings, args.llm, args.model)
    question = args.question or "公司年假有几天？"

    print("\n" + "=" * 64)
    print(f"问题：{question}")
    print("-" * 64 + "\n命中片段：")
    for i, doc in enumerate(retriever.invoke(question), 1):
        preview = doc.page_content.replace("\n", " ")[:60]
        print(f"  [{i}] {doc.metadata.get('source')}  {preview}...")
    print("-" * 64 + "\n答案：")
    print(chain.invoke(question))
    print("=" * 64)


if __name__ == "__main__":
    main()
```

> 小提示：上面只需 `langchain-core` + `langchain-text-splitters` 就能离线跑通（`--llm extractive`）；只有切到 `ollama` / `openai` 时才需要多装一个包。

**关键对照点**（左列是你在前几章亲手写过的调用）：

| 环节 | 本项目手写 | LangChain |
| --- | --- | --- |
| 切块 | `split_text(doc.text, 220, 50)` | `RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=50)` |
| 建库 | `store.add(vectors, texts, sources)` | `InMemoryVectorStore.from_documents(splits, embedding=embeddings)` |
| 检索 | `retriever.retrieve(q, top_k=3, min_score=0.2)` | `store.as_retriever(search_kwargs={"k": 3})` |
| 生成 | `self.generator.generate(q, contexts)` | `{"context": retriever \| format_docs, "question": RunnablePassthrough()} \| prompt \| llm \| StrOutputParser()` |
| 换向量库 | 重写 `vector_store.py` | 换 `FAISS` / `Chroma`，**一行** |
| 换大模型 | 新增一个类 + 改工厂 + 改命令行 | 换 `ChatOllama` / `ChatOpenAI`，**一行** |

### 4. 跑起来会看到什么

`--llm extractive`（默认）完全离线，装了最小组合就能跑：

```powershell
python langchain_demo.py -q "公司年假有几天？"
```

预期输出（示例；实际以你安装的版本与数据为准）：

```text
================================================================
问题：公司年假有几天？
----------------------------------------------------------------
命中片段：
  [1] company_handbook.md  # 星野科技 员工手册（节选） ## 一、年假政策 正式员工入职满 1 年可享受 5 天...
  [2] ...（其余命中片段，通常仍是年假所在文档及其邻近段落）
  [3] ...
----------------------------------------------------------------
答案：
关于「公司年假有几天？」，命中 3 条资料：

[1] (来源：company_handbook.md) # 星野科技 员工手册（节选）
## 一、年假政策
正式员工入职满 1 年可享受 5 天带薪年假；满 3 年增加到 10 天；满 5 年及以上增加到 15 天。
[2] ...（略）
[3] ...（略）
================================================================
```

> 为什么是"命中 3 条"？因为 `as_retriever(search_kwargs={"k": 3})` 取的就是 `Settings.top_k`（默认 3），
> 而且**框架默认不过滤低分**——这点和手写版 `retriever.retrieve(..., min_score=0.2)` 不同：手写版会把
> 相似度低于 0.2 的噪声块直接丢掉。想对齐，可把 `min_score` 交给 LangChain 的 `search_kwargs`（如 `score_threshold`）。

换成 `--llm ollama` / `--llm openai` 时，末尾那段会变成大模型组织过的通顺答案（相当于 `python main.py --generator ollama` 的效果）。

### 5. 效果对比：LangChain 到底改变了什么

| 维度 | 本项目（手写） | LangChain 版 | 谁更优 |
| --- | --- | --- | --- |
| 检索命中 | 见 `python main.py -q "公司年假有几天？"` | 数据与向量化器完全相同，命中片段高度重合（差别主要来自切块器切出的边界不同） | 基本一致 |
| 主体代码量 | 7 个模块 | 1 个文件、约 120 行 | LangChain |
| 换向量模型 | 改 `build_embedder`，可能要动多处 | 换成 `OpenAIEmbeddings(...)` 等，**一行** | LangChain |
| 换向量库 | 重写 `vector_store.py` | 换成 `FAISS` / `Chroma`，**一行** | LangChain |
| 换大模型 | 新增类 + 改工厂 + 改 CLI | 换成 `ChatOllama` / `ChatOpenAI`，**一行** | LangChain |
| 生态 | 无 | 数百种 loader / store / model 集成 | LangChain |
| 可观测性 | `print` 到底 | 可接 LangSmith 追踪整个链路 | LangChain |
| 可控性 / 可调试 | 每一行都懂，出错好定位 | 抽象层多，出错往往要读框架源码 | 手写（学习期） |
| 依赖体积 | **只有 `numpy`** | 若干包 + 一堆传递依赖 | 手写 |

**三条结论：**

1. **检索质量 = 数据质量 × 切块策略 × 向量化模型**，跟你用不用 LangChain **无关**。本章刻意"只换编排、不换向量化"，就是为了说明这一点——两边能查到的东西本质相同，因为底层零件一模一样。
2. LangChain 的真正价值在**"换零件"的成本**上：从 TF-IDF 换到 BGE、从内存库换到 FAISS、从 Ollama 换到 GPT，用框架通常只要改一行；手写则要动好几个文件。
3. 所以推荐姿势是：**学原理用手写，做产品 / 快速试错用 LangChain。** 两者不矛盾——你先手写一遍，正是为了看懂 LangChain 那"一行"背后到底发生了什么。

### 6. 什么时候该用 / 不该用

**该用：**

- 要接很多"零件"（多种向量库 / 多种模型 / 多种文档加载器）；
- 需要流式输出、工具调用、Agent、可观测性（LangSmith）；
- 快速做原型、验证想法。

**不该用：**

- 只是想**理解 RAG**——抽象层会挡住你观察"检索到底准不准"的视线；
- 追求**轻依赖 / 小体积部署**；
- 需求很固定（就一个库、一个模型），手写反而更可控、更好排错。

### 7. 常见坑

| 坑 | 说明 | 对策 |
| --- | --- | --- |
| 包拆得太碎 | `langchain` 已拆成 `langchain-core` / `langchain-community` / `langchain-<厂商>` | 按需安装；导入路径以实际版本为准 |
| API 变动快 | 类名、路径会随大版本迁移 | 锁定版本；查你所装版本的官方 API 文档 |
| 中文切块不理想 | `RecursiveCharacterTextSplitter` 默认分隔符对中文标点不友好 | 在 `separators` 里显式加 `"。"` `"！"` `"？"` |
| TF-IDF 适配陷阱 | `Embeddings` 接口是"无状态"的，而 TF-IDF 必须先 `fit` | 顺序必须是"先 `embed_documents` 建词表，再 `embed_query`"；增量入库需重新 `fit` |
| 相似度分数不可比 | LangChain 向量库的打分口径（余弦 / 欧氏）与手写版不同 | 只比较"命中了哪几段"，别直接比分数绝对值 |
| 缺包报错 | `ModuleNotFoundError: No module named 'langchain_ollama'` | 对应执行 `pip install langchain-ollama`（其余同理） |

---

📚 更详细的原理与"从零手写"的叙述，见 [`docs/TUTORIAL.md`](docs/TUTORIAL.md)。

