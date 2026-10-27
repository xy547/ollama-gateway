"""外挂二：本地知识库检索（RAG 的 R）。

刻意**不引入向量库和 embedding 模型**，原因有两个：

1. 你机器上可能只有一个小 chat 模型，拉 embedding 模型要多下几百 MB；
2. BM25 在"私有文档问答"这类场景召回并不差，而且零依赖、可解释、毫秒级。

中文分词用 **字符 bigram**（"梯度下降" -> 梯/度/下/降/梯度/度下/下降），
这是无词典场景下最省事又不太差的做法；英文按词切。

要换成向量检索也简单：实现同样的 ``search(query, top_k)`` 接口替换 ``get_store()`` 即可。
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import settings

CJK = re.compile(r"[\u4e00-\u9fff]")
WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.-]*")
SUPPORTED = {".md", ".txt", ".markdown"}


def tokenize(text: str, for_query: bool = False) -> list[str]:
    """中英混合分词：英文按词，中文按字 + 相邻 bigram。

    ``for_query=True`` 时，长度 ≥2 的中文片段**只产出 bigram**。
    这是被测试逼出来的修正：查询"量子纠缠退相干"如果按单字切，会命中
    文档里的"相关性"的"相"，得一个假分。bigram 才能表达"相邻才算匹配"。
    单字片段（"的""是"）仍保留，否则"猫"这种单字查询永远搜不到东西。
    """
    text = text.lower()
    tokens = [w for w in WORD.findall(text)]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if for_query and len(run) >= 2:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.extend(run)  # unigram
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigram
    return tokens


def split_chunks(text: str, size: int = 420, overlap: int = 60) -> list[str]:
    """按段落聚合切块，带一点重叠，避免答案被切断。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= size:
                buf = p
            else:
                # 超长段落硬切
                start = 0
                while start < len(p):
                    chunks.append(p[start : start + size])
                    start += size - overlap
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


class BM25Store:
    """一个极简 BM25 实现（k1=1.5, b=0.75）。"""

    def __init__(self, docs_dir: Path | None = None):
        self.docs_dir = docs_dir or (Path(settings.data_path) / "docs")
        self._lock = threading.Lock()
        self._chunks: list[dict[str, Any]] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        self._stamp: float = -1
        self._loaded = False

    # ------------------------------------------------------------ 索引
    def _sig(self) -> float:
        if not self.docs_dir.exists():
            return 0.0
        return max((p.stat().st_mtime for p in self.docs_dir.rglob("*") if p.is_file()), default=0.0)

    def build(self, force: bool = False) -> None:
        with self._lock:
            sig = self._sig()
            if self._loaded and not force and sig == self._stamp:
                return
            chunks: list[dict[str, Any]] = []
            df: Counter = Counter()
            if self.docs_dir.exists():
                for path in sorted(self.docs_dir.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                        continue
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    for i, chunk in enumerate(split_chunks(text)):
                        toks = tokenize(chunk)
                        if not toks:
                            continue
                        tf = Counter(toks)
                        chunks.append(
                            {
                                "source": str(path.relative_to(self.docs_dir)).replace("\\", "/"),
                                "chunk_index": i,
                                "text": chunk,
                                "tf": tf,
                                "len": len(toks),
                            }
                        )
                        df.update(tf.keys())
            self._chunks = chunks
            self._df = df
            self._avgdl = (sum(c["len"] for c in chunks) / len(chunks)) if chunks else 0.0
            self._stamp = sig
            self._loaded = True

    # ------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        self.build()
        if not self._chunks:
            return []
        k1, b = 1.5, 0.75
        n = len(self._chunks)
        q_tokens = Counter(tokenize(query, for_query=True))
        scored: list[tuple[float, dict[str, Any]]] = []
        for c in self._chunks:
            score = 0.0
            for term, qf in q_tokens.items():
                f = c["tf"].get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += (
                    qf * idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * c["len"] / (self._avgdl or 1)))
                )
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"source": c["source"], "chunk_index": c["chunk_index"], "text": c["text"], "score": s}
            for s, c in scored[:top_k]
        ]

    # ------------------------------------------------------------ 维护
    def stats(self) -> dict[str, Any]:
        self.build()
        files = {c["source"] for c in self._chunks}
        return {
            "documents": len(files),
            "chunks": len(self._chunks),
            "vocab": len(self._df),
            "docs_dir": str(self.docs_dir),
        }

    def add_document(self, filename: str, content: str) -> dict[str, Any]:
        """写入一篇文档并立即重建索引。"""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        safe = Path(filename).name  # 防目录穿越
        if not safe.endswith((".md", ".txt")):
            safe += ".md"
        (self.docs_dir / safe).write_text(content, encoding="utf-8")
        self.build(force=True)
        return {"saved": safe, "chars": len(content), **self.stats()}


_store: BM25Store | None = None
_store_lock = threading.Lock()


def get_store() -> BM25Store:
    global _store
    with _store_lock:
        if _store is None:
            _store = BM25Store()
        return _store


def build_rag_context(query: str, top_k: int = 3) -> tuple[str, list[dict[str, Any]]]:
    """把检索结果拼成一段可直接塞进 system 的上下文。"""
    hits = get_store().search(query, top_k=top_k)
    if not hits:
        return "", []
    lines = ["以下是从本地知识库检索到的资料，回答时优先依据它们，并在末尾标注来源文件名："]
    for i, h in enumerate(hits, 1):
        lines.append(f"\n[资料{i}｜来源：{h['source']}]\n{h['text']}")
    return "\n".join(lines), hits
