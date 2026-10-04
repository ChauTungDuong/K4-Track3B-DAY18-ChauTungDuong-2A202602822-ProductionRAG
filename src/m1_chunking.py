from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


_SEMANTIC_MODEL = None


def _get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    from numpy import dot
    from numpy.linalg import norm

    import torch
    model = _get_semantic_model()
    with torch.inference_mode():
        embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False)

    def _sim(a, b):
        denom = norm(a) * norm(b)
        return float(dot(a, b) / (denom + 1e-9)) if denom > 0 else 0.0

    groups = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = _sim(embeddings[i - 1], embeddings[i])
        if similarity < threshold:
            groups.append(" ".join(current_group))
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])
    if current_group:
        groups.append(" ".join(current_group))

    return [
        Chunk(text=group, metadata={**metadata, "strategy": "semantic", "chunk_index": idx})
        for idx, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        return ([], [])

    paras = []
    for p in raw_paras:
        if len(p) <= parent_size:
            paras.append(p)
        else:
            for i in range(0, len(p), parent_size):
                sub = p[i:i + parent_size].strip()
                if sub:
                    paras.append(sub)

    parents: list[Chunk] = []
    current_parent = ""
    for p in paras:
        if len(current_parent) + len(p) + 2 > parent_size and current_parent:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current_parent.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current_parent = ""
        current_parent = f"{current_parent}\n\n{p}".strip() if current_parent else p

    if current_parent.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current_parent.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        p_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', parent.text) if s.strip()]
        cur_child = ""
        for s in p_sentences:
            if len(s) > child_size:
                if cur_child:
                    children.append(Chunk(
                        text=cur_child.strip(),
                        metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                        parent_id=pid
                    ))
                    cur_child = ""
                for j in range(0, len(s), child_size):
                    sub = s[j:j + child_size].strip()
                    if sub:
                        children.append(Chunk(
                            text=sub,
                            metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                            parent_id=pid
                        ))
            elif len(cur_child) + len(s) + 1 > child_size and cur_child:
                children.append(Chunk(
                    text=cur_child.strip(),
                    metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                    parent_id=pid
                ))
                cur_child = s
            else:
                cur_child = f"{cur_child} {s}".strip() if cur_child else s

        if cur_child.strip():
            children.append(Chunk(
                text=cur_child.strip(),
                metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                parent_id=pid
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    chunks = []
    current_header = ""
    current_content = ""

    for sec in sections:
        sec_stripped = sec.strip()
        if not sec_stripped:
            continue
        if re.match(r'^#{1,3}\s+', sec_stripped):
            if current_header or current_content:
                full_text = f"{current_header}\n\n{current_content}".strip() if current_header else current_content.strip()
                if full_text:
                    meta = {**metadata, "strategy": "structure"}
                    if current_header:
                        meta["section"] = re.sub(r'^#{1,3}\s+', '', current_header).strip()
                    chunks.append(Chunk(text=full_text, metadata=meta))
            current_header = sec_stripped
            current_content = ""
        else:
            current_content = f"{current_content}\n\n{sec_stripped}".strip() if current_content else sec_stripped

    if current_header or current_content:
        full_text = f"{current_header}\n\n{current_content}".strip() if current_header else current_content.strip()
        if full_text:
            meta = {**metadata, "strategy": "structure"}
            if current_header:
                meta["section"] = re.sub(r'^#{1,3}\s+', '', current_header).strip()
            chunks.append(Chunk(text=full_text, metadata=meta))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
