"""
Semantic Matcher — connects Qdrant to the LLM classifier and arbitrage calculator.

Flow:
    Qdrant (similar contracts) → LLM (classify relationship) → ArbitrageCalculator

Run:
    python semantic_matcher.py
    python semantic_matcher.py --top-k 5 --threshold 0.80
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

from arbitrage_calculator import ArbitrageCalculator, ArbitrageOpportunity
from models import UnifiedMarket, Exchange, MarketStatus, OutcomeType, ResolutionSource

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "markets"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"

SIMILARITY_THRESHOLD = 0.82
TOP_K    = 5
MIN_ROI  = 0.005


@dataclass
class ClassifiedPair:
    market_a_uid:  str
    market_b_uid:  str
    similarity:    float
    relation:      str
    confidence:    float
    reasoning:     str
    opportunities: list[ArbitrageOpportunity]


SYSTEM_PROMPT = """You are a prediction market analyst. Given two prediction market contracts,
classify their relationship. Reply ONLY with valid JSON — no explanation, no markdown.

Relationships:
- IDENTICAL: Both contracts resolve YES for the exact same real-world event.
- COMPLEMENT: One contract is the logical opposite of the other (A:YES == B:NO).
- SUBSET: One contract is a narrower version of the other.
- MUTUALLY_EXCLUSIVE: Both can't resolve YES at the same time, but aren't complements.
- UNRELATED: No meaningful relationship.

Required JSON format:
{
  "relation": "IDENTICAL",
  "confidence": 0.95,
  "reasoning": "one sentence max"
}"""


def classify_pair(
    client: Groq,
    text_a: str,
    text_b: str,
    exchange_a: str,
    exchange_b: str,
) -> tuple[str, float, str]:
    """Send a pair to the LLM and get back a relationship label."""
    user_msg = (
        f"Contract A ({exchange_a}):\n{text_a}\n\n"
        f"Contract B ({exchange_b}):\n{text_b}"
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=120,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        return (
            parsed.get("relation", "UNRELATED").upper(),
            float(parsed.get("confidence", 0.0)),
            parsed.get("reasoning", ""),
        )

    except (json.JSONDecodeError, KeyError):
        return "UNRELATED", 0.0, "parse error"
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "UNRELATED", 0.0, str(e)


def payload_to_market(payload: dict, uid: str) -> UnifiedMarket | None:
    """Reconstruct a minimal UnifiedMarket from a Qdrant payload dict."""
    try:
        return UnifiedMarket(
            exchange=Exchange(payload.get("exchange", "kalshi")),
            native_id=payload.get("native_id", uid),
            question=payload.get("question", ""),
            description=payload.get("description", ""),
            yes_price=payload.get("yes_price"),
            no_price=payload.get("no_price"),
            status=MarketStatus(payload.get("status", "unknown")),
            outcome_type=OutcomeType(payload.get("outcome_type", "binary")),
            resolution_source=ResolutionSource(
                payload.get("resolution_source", "unknown")
            ),
        )
    except Exception as e:
        logger.debug(f"Could not reconstruct market {uid}: {e}")
        return None


def run_matcher(
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    min_roi: float = MIN_ROI,
    fee_rate: float = 0.02,
) -> list[ClassifiedPair]:

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    groq_client = Groq(api_key=groq_key)
    qdrant      = QdrantClient(path="qdrant_data")
    qdrant.set_model(EMBED_MODEL)
    calc        = ArbitrageCalculator(default_fee_rate=fee_rate)

    # ── 1. Load all markets from Qdrant ──────────────────────────────────────
    logger.info("Loading all markets from Qdrant...")
    all_points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    logger.info(f"Loaded {len(all_points)} markets")

    if len(all_points) < 2:
        logger.warning("Need at least 2 markets. Run pipeline.py --vector first.")
        return []

    # ── 2. Find candidate pairs via similarity search ─────────────────────────
    seen_pairs: set[frozenset] = set()
    candidates: list[tuple]   = []

    for point in all_points:
        payload = point.payload or {}
        uid_a   = payload.get("uid", str(point.id))
        text_a  = payload.get("embedding_text") or payload.get("question", "")

        if not text_a:
            continue

        results = qdrant.query(
            collection_name=COLLECTION_NAME,
            query_text=text_a,
            limit=top_k + 1,
        )

        for hit in results:
            hit_payload = hit.metadata if hasattr(hit, "metadata") else hit.payload
            uid_b       = hit_payload.get("uid", str(hit.id))

            if uid_a == uid_b:
                continue
            if hit.score < threshold:
                continue

            pair_key = frozenset([uid_a, uid_b])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            candidates.append((point, hit, hit.score))

    logger.info(f"Found {len(candidates)} candidate pairs above threshold {threshold}")

    # ── 3. Classify each pair with the LLM ───────────────────────────────────
    results_out: list[ClassifiedPair] = []

    for i, (pt_a, pt_b, similarity) in enumerate(candidates):
        payload_a  = pt_a.payload or {}
        payload_b  = pt_b.metadata if hasattr(pt_b, "metadata") else pt_b.payload or {}

        uid_a      = payload_a.get("uid", str(pt_a.id))
        uid_b      = payload_b.get("uid", str(pt_b.id))
        exchange_a = payload_a.get("exchange", "unknown")
        exchange_b = payload_b.get("exchange", "unknown")
        text_a     = payload_a.get("embedding_text") or payload_a.get("question", "")
        text_b     = payload_b.get("embedding_text") or payload_b.get("question", "")

        logger.info(
            f"[{i+1}/{len(candidates)}] ({exchange_a}) vs ({exchange_b}) "
            f"| sim={similarity:.3f}"
        )

        relation, confidence, reasoning = classify_pair(
            groq_client, text_a, text_b, exchange_a, exchange_b
        )

        # ── 4. Run arb calculator on actionable pairs ─────────────────────
        opportunities: list[ArbitrageOpportunity] = []

        if relation in ("IDENTICAL", "COMPLEMENT") and confidence >= 0.80:
            market_a = payload_to_market(payload_a, uid_a)
            market_b = payload_to_market(payload_b, uid_b)

            if market_a and market_b:
                opps = calc.evaluate_pair(market_a, market_b, relation)
                opportunities = [o for o in opps if o.roi >= min_roi]

        results_out.append(ClassifiedPair(
            market_a_uid=uid_a,
            market_b_uid=uid_b,
            similarity=similarity,
            relation=relation,
            confidence=confidence,
            reasoning=reasoning,
            opportunities=opportunities,
        ))

        time.sleep(0.1)  # avoid hammering Groq rate limits

    # ── 5. Print results ──────────────────────────────────────────────────────
    arb_pairs = [r for r in results_out if r.opportunities]
    logger.info(f"{len(results_out)} pairs classified, {len(arb_pairs)} with arb opportunities")

    for pair in arb_pairs:
        print("\n" + "=" * 70)
        print(f"PAIR:      {pair.market_a_uid}  ↔  {pair.market_b_uid}")
        print(f"RELATION:  {pair.relation} (confidence={pair.confidence:.2f})")
        print(f"REASONING: {pair.reasoning}")
        print(f"SIM SCORE: {pair.similarity:.4f}")
        for opp in pair.opportunities:
            print(f"\n  {opp}")
        print("=" * 70)

    return results_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic market matcher")
    parser.add_argument("--top-k",     type=int,   default=TOP_K,                help="Qdrant neighbours per market")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Min similarity (0-1)")
    parser.add_argument("--min-roi",   type=float, default=MIN_ROI,              help="Min ROI to report")
    parser.add_argument("--fee-rate",  type=float, default=0.02,                 help="Trading fee rate")
    args = parser.parse_args()

    run_matcher(
        top_k=args.top_k,
        threshold=args.threshold,
        min_roi=args.min_roi,
        fee_rate=args.fee_rate,
    )