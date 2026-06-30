from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import os, sys, re
from collections import Counter
from dataclasses import dataclass
from math import log, sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
    except (ImportError, RuntimeError):
        segmented = text
    return " ".join(re.findall(r"\w+", segmented.replace("_", " ").lower(), re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c.get("text", "")).split() for c in chunks]
        self.bm25 = None
        if self.corpus_tokens:
            try:
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi(self.corpus_tokens)
            except Exception as exc:
                print(f"  ⚠️  rank-bm25 unavailable; using compatible local BM25: {exc}")
                self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=dict(self.documents[i].get("metadata", {})),
                method="bm25",
            )
            for i in top_indices[:top_k]
            if float(scores[i]) > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=2,
            check_compatibility=False,
        )
        self._encoder = None
        self._local_collections: dict[str, list[dict]] = {}
        self._qdrant_collections: set[str] = set()

    def _get_encoder(self):
        if self._encoder is None:
            try:
                if os.getenv("RAG_USE_TRANSFORMERS", "").lower() not in {"1", "true", "yes"}:
                    raise RuntimeError("transformer embeddings disabled for fast/offline execution")
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(
                    EMBEDDING_MODEL,
                    model_kwargs={"local_files_only": True},
                )
            except Exception:
                self._encoder = _HashingEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        texts = [c.get("text", "") for c in chunks]
        if not texts:
            self._local_collections[collection] = []
            return
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        records = [
            {
                "text": text,
                "metadata": dict(chunk.get("metadata", {})),
                "vector": _as_list(vector),
            }
            for text, chunk, vector in zip(texts, chunks, vectors)
        ]
        self._local_collections[collection] = records

        try:
            if os.getenv("RAG_USE_QDRANT", "").lower() not in {"1", "true", "yes"}:
                raise RuntimeError("Qdrant disabled; set RAG_USE_QDRANT=1 to enable")
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=len(records[0]["vector"]), distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=record["vector"],
                    payload={**record["metadata"], "text": record["text"]},
                )
                for i, record in enumerate(records)
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)
            self._qdrant_collections.add(collection)
        except Exception as exc:
            print(f"  ⚠️  Qdrant unavailable; using in-process dense index: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        query_vector = _as_list(self._get_encoder().encode(query))
        if collection in self._qdrant_collections:
            try:
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                return [
                    SearchResult(
                        text=(point.payload or {}).get("text", ""),
                        score=float(point.score),
                        metadata={k: v for k, v in (point.payload or {}).items() if k != "text"},
                        method="dense",
                    )
                    for point in response.points
                ]
            except Exception as exc:
                print(f"  ⚠️  Qdrant query failed; using local dense search: {exc}")

        scored = []
        for record in self._local_collections.get(collection, []):
            scored.append((_cosine(query_vector, record["vector"]), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(record["text"], float(score), dict(record["metadata"]), "dense")
            for score, record in scored[:top_k]
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if top_k <= 0:
        return []
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked[:top_k]
    ]


def _as_list(vector) -> list[float]:
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm + 1e-12)


class _HashingEncoder:
    """Dependency-free normalized bag-of-words encoder for offline execution."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in segment_vietnamese(text).split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
        norm = sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            return self._encode_one(texts)
        return [self._encode_one(text) for text in texts]


class _SimpleBM25:
    """Small BM25Okapi-compatible fallback used when NumPy cannot be imported."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in corpus]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self.frequencies = [Counter(document) for document in corpus]
        document_frequency = Counter()
        for document in corpus:
            document_frequency.update(set(document))
        count = len(corpus)
        self.idf = {
            token: log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for frequencies, doc_len in zip(self.frequencies, self.lengths):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
