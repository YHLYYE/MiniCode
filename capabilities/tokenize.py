"""双语关键词分词 —— 全项目唯一一份。

为什么抽出来：有两处都要做「中英混合文本 → 关键词集合」，但原本各写一份，
而且 memory 那份对中文是**坏的**：它按 `[\\W_]+` 切分，而 Python 的 `\\w`
**包含中文**，于是中文整句不会被切开 —— 实测
`_tokenize("蜂巢取快递验证码摁错怎么办")` 只得到 1 个 token（整句），
与语料的交集恒为空，**中文程序性记忆永远搜不出来**（英文路径正常）。

和之前「两份危险命令正则合并成一份」是同一个教训：同一件事只留一处实现。
（skill 那边对中文是对的：字符 bigram。但它和 memory 各写一份，谁对谁错没人对。）

规则：
- 英文/数字：`[a-zA-Z0-9]+` 取词并小写。**不在下划线处保留整词**——记忆里存的是
  代码，标识符拆开更好召回（`record_file_edit` 要能被 "file edit" 搜到）。
  这条与 RAG 项目 `src/rag/bm25.py::simple_tokenize` 同一规则。
- CJK 连续段：字符**二元组**（单字段落取单字）。不引词典、不引依赖，
  与 Lucene CJKAnalyzer 和本项目 RAG 侧的 `zh_tokenize` 同一思路。

为什么不用 jieba：多一个依赖、多一份词典，而且检索场景 char bigram 已经够用
（RAG 项目实测中文 BM25 命中率 0.08 → 0.9967）。
"""
import re

_ENGLISH_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def keyword_tokens(text: str | None) -> set[str]:
    """把一段中英混合文本切成关键词集合。"""
    text = text or ""
    tokens: set[str] = {w.lower() for w in _ENGLISH_RE.findall(text)}

    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i:i + 2] for i in range(len(run) - 1))

    return tokens
