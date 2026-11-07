"""Chunking, embedding and Qdrant storage."""
from __future__ import annotations
import os, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.feature_extraction.text import HashingVectorizer
from .tracing import observe


@dataclass
class Chunk:
    id: int
    source: str      # file#section, shown to the user as citation
    text: str


def chunk_markdown(path: Path) -> list[tuple[str, str]]:
    """One chunk per '##' section, title prepended so the section stays self-explanatory."""
    text = path.read_text(encoding="utf-8")
    doc_title = re.search(r"^# (.+)$", text, re.M).group(1)
    out = []
    for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        section, body = m.group(1).strip(), " ".join(m.group(2).split())
        out.append((f"{path.name}#{section}", f"{doc_title} – {section}: {body}"))
    return out


class HashingEmbedder:
    """Offline default: deterministic, no API key. Swap for a neural embedder in production."""
    dim = 2048

    def __init__(self) -> None:
        self.v = HashingVectorizer(n_features=self.dim, alternate_sign=False, norm="l2",
                                   ngram_range=(1, 2), stop_words="english")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.v.transform(texts).toarray()


class OpenAIEmbedder:
    dim = 1536

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in r.data])


def default_embedder():
    return OpenAIEmbedder() if os.getenv("OPENAI_API_KEY") else HashingEmbedder()


class Index:
    def __init__(self, embedder=None, url: str | None = None, collection: str = "docs") -> None:
        self.emb = embedder or default_embedder()
        self.db = QdrantClient(url=url) if url else QdrantClient(":memory:")
        self.col, self.chunks = collection, {}

    def build(self, folder: Path) -> int:
        pairs = [p for f in sorted(folder.glob("*.md")) for p in chunk_markdown(f)]
        vecs = self.emb.embed([t for _, t in pairs])
        if self.db.collection_exists(self.col):
            self.db.delete_collection(self.col)
        self.db.create_collection(self.col, vectors_config=VectorParams(size=self.emb.dim, distance=Distance.COSINE))
        points = []
        for i, ((src, text), v) in enumerate(zip(pairs, vecs)):
            self.chunks[i] = Chunk(i, src, text)
            points.append(PointStruct(id=i, vector=v.tolist(), payload={"source": src, "text": text}))
        self.db.upsert(self.col, points=points)
        return len(points)

    @observe(name="retrieve")
    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        v = self.emb.embed([query])[0].tolist()
        hits = self.db.query_points(self.col, query=v, limit=k).points
        # text comes from the vector store payload, so a persistent Qdrant server works after a restart
        return [(Chunk(h.id, h.payload["source"], h.payload["text"]), h.score) for h in hits]
