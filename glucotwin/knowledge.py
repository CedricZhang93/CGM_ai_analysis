"""Long-term memory (RAG) built from scratch: chunk -> embed -> store in SQLite -> cosine search.

Replaces LangChain + ChromaDB with ~100 lines and numpy.
"""

import io
import re

import numpy as np

from glucotwin import db
from glucotwin.llm import EmbeddingModel


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """Split text into chunks on paragraph/sentence boundaries, with a small overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    # Paragraphs first, then sentences for paragraphs that are still too long
    pieces = []
    for para in text.split("\n\n"):
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            pieces.extend(s for s in re.split(r"(?<=[。！？!?.；;\n])", para) if s.strip())

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > chunk_size:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += ("\n\n" if current else "") + piece
        # Hard split for a single piece that exceeds chunk_size on its own
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            current = current[chunk_size - overlap :]
    if current.strip():
        chunks.append(current.strip())
    return chunks


def read_document(content: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("解析 PDF 需要安装 pypdf: pip install pypdf") from e
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


class KnowledgeBase:
    def __init__(self, embedder: EmbeddingModel):
        self.embedder = embedder

    def add_document(self, content: bytes, filename: str) -> int:
        """Chunk, embed and store a document. Re-uploading the same filename replaces it."""
        chunks = chunk_text(read_document(content, filename))
        if not chunks:
            raise ValueError("文档为空")
        vectors = self.embedder.embed(chunks)
        rows = [
            (filename, chunk, np.asarray(vec, dtype=np.float32).tobytes())
            for chunk, vec in zip(chunks, vectors)
        ]
        with db.connect() as conn:
            conn.execute("DELETE FROM kb_chunks WHERE source=?", (filename,))
            conn.executemany("INSERT INTO kb_chunks (source,text,embedding) VALUES (?,?,?)", rows)
        return len(rows)

    def search(self, query: str, k: int = 3, min_score: float = 0.3) -> list[dict]:
        """Return the top-k chunks by cosine similarity."""
        with db.connect() as conn:
            rows = conn.execute("SELECT source,text,embedding FROM kb_chunks").fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        q = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        scores = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-10)
        top = np.argsort(-scores)[:k]
        return [
            {"source": rows[i][0], "text": rows[i][1], "score": round(float(scores[i]), 3)}
            for i in top
            if scores[i] >= min_score
        ]

    @staticmethod
    def stats() -> dict:
        with db.connect() as conn:
            n, files = conn.execute("SELECT COUNT(*), COUNT(DISTINCT source) FROM kb_chunks").fetchone()
        return {"chunks": n, "files": files}
