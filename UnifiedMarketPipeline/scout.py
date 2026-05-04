"""
Scout: Automated Market Matcher

This script performs the "Discovery" phase:
1. Loops through markets on a source exchange (e.g., Kalshi).
2. Searches the vector store for similar markets on other exchanges.
3. Uses the LLM Classifier to verify the relationship.
4. Saves confirmed matches to verified_pairs.json.
"""

import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from models import UnifiedMarket, Exchange, Outcome
from qdrant_client import models
from vector_store import setup_client, COLLECTION_NAME
from llm_classifier import MarketClassifier

logger = logging.getLogger("scout")

STORAGE_FILE = "verified_pairs.json"
DEFAULT_THRESHOLD = 0.55
DEFAULT_LLM_THRESHOLD = 0.80
DEFAULT_ACCEPT_LABELS: Set[str] = {"IDENTICAL", "COMPLEMENT"}


class Scout:
    def __init__(
        self,
        dry_run: bool = False,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        llm_threshold: float = DEFAULT_LLM_THRESHOLD,
        qdrant_path: Optional[str] = None,
        collection: Optional[str] = None,
        storage_file: str = STORAGE_FILE,
        accept_labels: Optional[Set[str]] = None,
        min_confidence: float = 0.7,
        llm_call_delay: float = 0.0,
    ):
        self.qdrant = setup_client(path=qdrant_path, collection=collection)
        self.collection = collection or COLLECTION_NAME
        self.classifier = None if dry_run else MarketClassifier()
        self.dry_run = dry_run
        self.similarity_threshold = similarity_threshold
        self.llm_threshold = llm_threshold
        self.storage_file = storage_file
        self.accept_labels = accept_labels if accept_labels is not None else DEFAULT_ACCEPT_LABELS
        self.min_confidence = min_confidence
        self.llm_call_delay = llm_call_delay
        self.verified_pairs: List[Dict] = self._load_storage()

    def _load_storage(self) -> List[Dict]:
        path = Path(self.storage_file)
        if not path.exists():
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            raise
        except OSError:
            return []

    def _save_storage(self):
        with open(self.storage_file, "w") as f:
            json.dump(self.verified_pairs, f, indent=2, default=str)
        logger.info(f"Saved {len(self.verified_pairs)} pairs to {self.storage_file}")

    def find_matches(
        self,
        source_exchange: Exchange,
        limit: int = 200,
        candidates_per_source: int = 10,
    ):
        """Main loop to find matches for markets from a specific exchange."""
        logger.info(
            f"Starting scout for {source_exchange.value} "
            f"(limit={limit}, candidates_per_source={candidates_per_source}, "
            f"threshold={self.similarity_threshold})..."
        )

        # 1. Scroll source markets from Qdrant
        source_points, _ = self.qdrant.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="exchange",
                        match=models.MatchValue(value=source_exchange.value),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        if not source_points:
            logger.warning(f"No markets found in collection '{self.collection}' for {source_exchange.value}.")
            return

        logger.info(f"Scanning {len(source_points)} {source_exchange.value} markets...")

        for point in source_points:
            market_a = UnifiedMarket(**point.payload)
            logger.info(f"Checking: [{market_a.exchange.value.upper()}] {market_a.question[:60]}...")

            # 2. Vector search for candidates on OTHER exchanges
            results = self.qdrant.query(
                collection_name=self.collection,
                query_text=market_a.embedding_text,
                query_filter=models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key="exchange",
                            match=models.MatchValue(value=source_exchange.value),
                        )
                    ]
                ),
                limit=candidates_per_source,
            )

            for result in results:
                if result.score < self.similarity_threshold:
                    continue

                market_b = UnifiedMarket(**result.metadata)
                logger.info(
                    f"  - Candidate (score={result.score:.4f}): "
                    f"[{market_b.exchange.value.upper()}] {market_b.question[:60]}"
                )

                if self._pair_exists(market_a.uid, market_b.uid):
                    logger.info("    (Already in watchlist, skipping LLM)")
                    continue

                if result.score < self.llm_threshold:
                    logger.info(
                        f"    (score={result.score:.4f} below LLM threshold "
                        f"{self.llm_threshold}, skipping classification)"
                    )
                    continue

                if self.dry_run:
                    logger.info("    (Dry run: skipping LLM classification)")
                    continue

                logger.info("    Calling LLM to verify relationship...")
                if self.llm_call_delay > 0:
                    time.sleep(self.llm_call_delay)
                hypothesis = self.classifier.classify_pair(market_a, market_b)
                logger.info(f"    Result: {hypothesis.label} (confidence={hypothesis.confidence:.2f})")

                if hypothesis.label in self.accept_labels and hypothesis.confidence >= self.min_confidence:
                    reason_lower = hypothesis.reason.lower()
                    reason_ok = bool(hypothesis.reason.strip()) and "unrelated" not in reason_lower
                    # For IDENTICAL/COMPLEMENT, reject reasons that describe contradicting conditions
                    if reason_ok and hypothesis.label in ("IDENTICAL", "COMPLEMENT"):
                        _contradictions = (
                            "not equivalent", "opposite", "inverse",
                            "not the same", "not identical", "however,",
                            "but ", "different direction", "different condition",
                        )
                        if any(p in reason_lower for p in _contradictions):
                            reason_ok = False
                    if reason_ok:
                        self._add_pair(market_a, market_b, hypothesis)
                    else:
                        logger.info(
                            f"    Rejected: reason contradicts label "
                            f"({hypothesis.reason[:80]!r})"
                        )

        self._save_storage()

    def _pair_exists(self, uid_a: str, uid_b: str) -> bool:
        for p in self.verified_pairs:
            uids = {p["market_a"]["uid"], p["market_b"]["uid"]}
            if uid_a in uids and uid_b in uids:
                return True
            # Block the same counterpart market from being paired with multiple source markets
            # (avoids pairing K threshold siblings with the same Polymarket question, and
            # vice versa when Polymarket is the source exchange)
            if uid_b in uids and uid_a not in uids:
                return True
            if uid_a in uids and uid_b not in uids:
                return True
        return False

    def _add_pair(self, m_a: UnifiedMarket, m_b: UnifiedMarket, hypothesis: Any):
        new_pair = {
            "pair_id": f"pair_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{m_a.native_id}",
            "market_a": self._market_dict(m_a),
            "market_b": self._market_dict(m_b),
            "relation": hypothesis.label,
            "confidence": hypothesis.confidence,
            "reason": hypothesis.reason,
            "discovered_at": datetime.now().isoformat(),
        }
        self.verified_pairs.append(new_pair)
        logger.info(f"ADDED: {m_a.question[:50]} <-> {m_b.question[:50]} ({hypothesis.label})")

    @staticmethod
    def _market_dict(m: UnifiedMarket) -> dict:
        """Serialize a market side with all fields needed by historical_data.py."""
        return {
            "uid": m.uid,
            "exchange": m.exchange.value,
            "native_id": m.native_id,
            "question": m.question,
            "yes_price": m.yes_price,
            "no_price": m.no_price,
            "url": m.url,
            "outcomes": [o.model_dump() for o in m.outcomes],
            "status": m.status.value,
            "result": m.result,
        }


def main():
    parser = argparse.ArgumentParser(description="Scout for semantic market matches.")
    parser.add_argument("--source", type=str, default="kalshi",
                        help="Exchange to use as reference (default: kalshi)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Number of source markets to scan (default: 200)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Vector similarity threshold for candidate retrieval (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--llm-threshold", type=float, default=DEFAULT_LLM_THRESHOLD,
                        help=f"Minimum score to escalate a candidate to the LLM (default: {DEFAULT_LLM_THRESHOLD})")
    parser.add_argument("--candidates", type=int, default=10,
                        help="Max candidates per source market (default: 10)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum LLM confidence to accept a pair (default: 0.7)")
    parser.add_argument("--accept-labels", type=str, default=None,
                        help="Comma-separated list of labels to accept (default: IDENTICAL,COMPLEMENT)")
    parser.add_argument("--qdrant-path", type=str, default=None,
                        help="Qdrant storage path (default: qdrant_data)")
    parser.add_argument("--collection", type=str, default=None,
                        help="Collection name (default: markets)")
    parser.add_argument("--storage-file", type=str, default=STORAGE_FILE,
                        help=f"Output pairs file (default: {STORAGE_FILE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform searches but skip LLM calls")

    args = parser.parse_args()

    accept_labels = (
        {label.strip().upper() for label in args.accept_labels.split(",") if label.strip()}
        if args.accept_labels else None
    )

    scout = Scout(
        dry_run=args.dry_run,
        similarity_threshold=args.threshold,
        llm_threshold=args.llm_threshold,
        qdrant_path=args.qdrant_path,
        collection=args.collection,
        storage_file=args.storage_file,
        accept_labels=accept_labels,
        min_confidence=args.min_confidence,
    )
    scout.find_matches(
        Exchange(args.source),
        limit=args.limit,
        candidates_per_source=args.candidates,
    )


if __name__ == "__main__":
    main()
