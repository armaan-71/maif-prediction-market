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
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from models import UnifiedMarket, Exchange
from qdrant_client import models
from qdrant_client import QdrantClient
from vector_store import setup_client, COLLECTION_NAME
from llm_classifier import MarketClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scout")

STORAGE_FILE = "verified_pairs.json"

class Scout:
    def __init__(self, dry_run: bool = False, similarity_threshold: float = 0.85):
        self.qdrant = setup_client()
        self.classifier = MarketClassifier()
        self.dry_run = dry_run
        self.similarity_threshold = similarity_threshold
        self.verified_pairs = self._load_storage()

    def _load_storage(self) -> List[Dict]:
        path = Path(STORAGE_FILE)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _save_storage(self):
        with open(STORAGE_FILE, "w") as f:
            json.dump(self.verified_pairs, f, indent=2, default=str)
        logger.info(f"Saved {len(self.verified_pairs)} pairs to {STORAGE_FILE}")

    def find_matches(self, source_exchange: Exchange, limit: int = 10):
        """
        Main loop to find matches for markets from a specific exchange.
        """
        logger.info(f"Starting scout for {source_exchange.value} (limit={limit})...")
        
        # 1. Fetch source markets from Qdrant
        source_points, _ = self.qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="exchange", match=models.MatchValue(value=source_exchange.value))
                ]
            ),
            limit=limit,
            with_payload=True
        )

        if not source_points:
            logger.warning(f"No markets found in Qdrant for {source_exchange.value}.")
            return

        for point in source_points:
            market_a = UnifiedMarket(**point.payload)
            logger.info(f"Checking: [{market_a.exchange.value.upper()}] {market_a.question[:60]}...")

            # 2. Vector Search for candidates on OTHER exchanges
            results = self.qdrant.query(
                collection_name=COLLECTION_NAME,
                query_text=market_a.embedding_text,
                query_filter=models.Filter(
                    must_not=[
                        models.FieldCondition(key="exchange", match=models.MatchValue(value=source_exchange.value))
                    ]
                ),
                limit=3
            )

            for result in results:
                if result.score < self.similarity_threshold:
                    continue
                
                market_b = UnifiedMarket(**result.metadata)
                logger.info(f"  - Potential Match (Score: {result.score:.4f}): [{market_b.exchange.value.upper()}] {market_b.question[:60]}")

                # Check if we already have this pair
                if self._pair_exists(market_a.uid, market_b.uid):
                    logger.info("    (Already in watchlist, skipping LLM)")
                    continue

                # 3. LLM Verification
                if self.dry_run:
                    logger.info("    (Dry run: skipping LLM classification)")
                    continue

                logger.info("    Calling LLM to verify relationship...")
                hypothesis = self.classifier.classify_pair(market_a, market_b)
                
                logger.info(f"    Result: {hypothesis.label} (Confidence: {hypothesis.confidence})")

                if hypothesis.label in ["IDENTICAL", "COMPLEMENT"] and hypothesis.confidence > 0.8:
                    self._add_pair(market_a, market_b, hypothesis)

        self._save_storage()

    def _pair_exists(self, uid_a: str, uid_b: str) -> bool:
        for p in self.verified_pairs:
            uids = {p["market_a"]["uid"], p["market_b"]["uid"]}
            if uid_a in uids and uid_b in uids:
                return True
        return False

    def _add_pair(self, m_a: UnifiedMarket, m_b: UnifiedMarket, hypothesis: Any):
        new_pair = {
            "pair_id": f"pair_{datetime.now().strftime('%Y%m%d%H%M%S')}_{m_a.native_id}",
            "market_a": {
                "uid": m_a.uid,
                "exchange": m_a.exchange.value,
                "question": m_a.question,
                "yes_price": m_a.yes_price
            },
            "market_b": {
                "uid": m_b.uid,
                "exchange": m_b.exchange.value,
                "question": m_b.question,
                "yes_price": m_b.yes_price
            },
            "relation": hypothesis.label,
            "confidence": hypothesis.confidence,
            "reason": hypothesis.reason,
            "discovered_at": datetime.now().isoformat()
        }
        self.verified_pairs.append(new_pair)
        logger.info(f"✅ ADDED TO WATCHLIST: {m_a.question} <-> {m_b.question} ({hypothesis.label})")

def main():
    parser = argparse.ArgumentParser(description="Scout for semantic market matches.")
    parser.add_argument("--source", type=str, default="kalshi", help="Exchange to use as reference")
    parser.add_argument("--limit", type=int, default=10, help="Number of markets to check")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
    parser.add_argument("--dry-run", action="store_true", help="Perform searches but skip LLM calls")
    
    args = parser.parse_args()
    
    scout = Scout(dry_run=args.dry_run, similarity_threshold=args.threshold)
    scout.find_matches(Exchange(args.source), limit=args.limit)

if __name__ == "__main__":
    main()
