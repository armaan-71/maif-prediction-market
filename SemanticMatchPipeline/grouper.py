"""
Qdrant KNN search + union-find to cluster semantically similar markets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from .models import MarketCluster, MarketSummary

COLLECTION_NAME = "markets"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_NAME = "fast-bge-small-en-v1.5"  # name FastEmbed assigns when using client.add()
DEFAULT_K = 5
DEFAULT_THRESHOLD = 0.78


def _setup_client(qdrant_path: str) -> QdrantClient:
    client = QdrantClient(path=qdrant_path)
    return client


def _embed(texts: list[str]) -> list[list[float]]:
    model = TextEmbedding(MODEL_NAME)
    return [vec.tolist() for vec in model.embed(texts)]


class _UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for k in self._parent:
            root = self.find(k)
            result.setdefault(root, []).append(k)
        return result


def build_clusters(
    markets: list[dict],
    qdrant_path: str = "qdrant_data",
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[MarketCluster]:
    """
    Query Qdrant for each market's nearest neighbors, then use union-find to
    form connected components. Returns only clusters with 2+ markets.
    """
    client = _setup_client(qdrant_path)

    try:
        client.get_collection(COLLECTION_NAME)
    except ValueError:
        print(
            f"[grouper] Collection '{COLLECTION_NAME}' not found. "
            "Run the UnifiedMarketPipeline with --vector first.",
            file=sys.stderr,
        )
        return []

    # Build uid → MarketSummary lookup
    market_by_uid: dict[str, MarketSummary] = {}
    for raw in markets:
        uid = raw.get("uid") or raw.get("native_id")
        if not uid:
            continue
        market_by_uid[uid] = MarketSummary(
            uid=uid,
            question=raw.get("question", ""),
            exchange=raw.get("exchange", ""),
            event_id=raw.get("event_id"),
            yes_price=raw.get("yes_price"),
            no_price=raw.get("no_price"),
            close_at=raw.get("close_at"),
            resolution_rules=raw.get("resolution_rules"),
            description=raw.get("description"),
            embedding_text=raw.get("embedding_text", raw.get("question", "")),
        )

    uids = list(market_by_uid.keys())
    summaries = list(market_by_uid.values())

    print(f"[grouper] Embedding {len(summaries)} markets...")
    vectors = _embed([s.embedding_text for s in summaries])
    uid_to_vector = {s.uid: v for s, v in zip(summaries, vectors)}

    uf = _UnionFind(uids)

    for uid, vector in uid_to_vector.items():
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            using=VECTOR_NAME,
            limit=k + 1,  # +1 because the market itself will appear
            with_payload=True,
        ).points
        for point in results:
            score = point.score
            if score < threshold:
                continue
            payload = point.payload or {}
            neighbor_uid = payload.get("uid") or payload.get("native_id")
            if not neighbor_uid or neighbor_uid == uid:
                continue
            if neighbor_uid in market_by_uid:
                uf.union(uid, neighbor_uid)

    groups = uf.groups()
    clusters: list[MarketCluster] = []
    cluster_id = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        clusters.append(
            MarketCluster(
                cluster_id=cluster_id,
                markets=[market_by_uid[m] for m in members if m in market_by_uid],
            )
        )
        cluster_id += 1

    return clusters


def load_markets(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"markets file not found: {path}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)
