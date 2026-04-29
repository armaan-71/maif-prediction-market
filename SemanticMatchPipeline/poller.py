"""
Price poller for discovered arbitrage candidates.

Reads arb_pairs.json (written by pipeline.py), fetches live quotes from each
exchange for the specific market IDs in each pair, recomputes the arb edge with
fresh prices, and logs any opportunities above the threshold.

Usage:
    python -m SemanticMatchPipeline.poller                    # poll once
    python -m SemanticMatchPipeline.poller --interval 30      # poll every 30s
    python -m SemanticMatchPipeline.poller --pairs SemanticMatchPipeline/arb_pairs.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

from .arbitrage import score_pair
from .models import ArbPairsOutput, ArbitrageCandidate, ClassifiedPair, MarketSummary, RelationType

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_BASE = "https://gamma-api.polymarket.com"
MANIFOLD_BASE = "https://api.manifold.markets/v0"


# ── Live price fetchers (one per exchange) ────────────────────────────────────

async def _kalshi_price(native_id: str, client: httpx.AsyncClient) -> tuple[Optional[float], Optional[float]]:
    resp = await client.get(f"{KALSHI_BASE}/markets/{native_id}")
    if resp.status_code != 200:
        return None, None
    m = resp.json().get("market", {})

    def _p(dollars_key: str, cents_key: str) -> Optional[float]:
        raw = m.get(dollars_key)
        if raw is not None:
            try:
                v = float(raw)
                return v if v > 0 else None
            except (ValueError, TypeError):
                pass
        raw = m.get(cents_key)
        if raw is not None:
            try:
                v = int(raw)
                return v / 100.0 if v > 0 else None
            except (ValueError, TypeError):
                pass
        return None

    yes = _p("yes_bid_dollars", "yes_bid")
    no = _p("no_bid_dollars", "no_bid")
    return yes, no


async def _polymarket_price(native_id: str, client: httpx.AsyncClient) -> tuple[Optional[float], Optional[float]]:
    # Gamma API requires conditionId as a query param, not a path segment
    resp = await client.get(f"{POLYMARKET_BASE}/markets", params={"conditionId": native_id})
    if resp.status_code != 200:
        return None, None
    data = resp.json()
    if not data:
        return None, None
    m = data[0]
    raw = m.get("outcomePrices", "[]")
    prices = json.loads(raw) if isinstance(raw, str) else raw
    yes = float(prices[0]) if prices else None
    no = float(prices[1]) if len(prices) > 1 else None
    return yes, no


async def _manifold_price(native_id: str, client: httpx.AsyncClient) -> tuple[Optional[float], Optional[float]]:
    resp = await client.get(f"{MANIFOLD_BASE}/market/{native_id}")
    if resp.status_code != 200:
        return None, None
    m = resp.json()
    prob = m.get("probability")
    if prob is None:
        return None, None
    return float(prob), round(1.0 - float(prob), 6)


async def fetch_live_price(
    exchange: str,
    native_id: str,
    client: httpx.AsyncClient,
) -> tuple[Optional[float], Optional[float]]:
    try:
        if exchange == "kalshi":
            return await _kalshi_price(native_id, client)
        if exchange == "polymarket":
            return await _polymarket_price(native_id, client)
        if exchange == "manifold":
            return await _manifold_price(native_id, client)
    except Exception as e:
        print(f"[poller] WARN: price fetch failed for {exchange}:{native_id} — {e}")
    return None, None


# ── Polling loop ──────────────────────────────────────────────────────────────

def _candidate_to_pair(
    candidate: ArbitrageCandidate,
    yes_a: Optional[float],
    no_a: Optional[float],
    yes_b: Optional[float],
    no_b: Optional[float],
) -> ClassifiedPair:
    return ClassifiedPair(
        market_a=MarketSummary(
            uid=candidate.uid_a,
            question=candidate.question_a,
            exchange=candidate.exchange_a,
            yes_price=yes_a,
            no_price=no_a,
            embedding_text=candidate.question_a,
        ),
        market_b=MarketSummary(
            uid=candidate.uid_b,
            question=candidate.question_b,
            exchange=candidate.exchange_b,
            yes_price=yes_b,
            no_price=no_b,
            embedding_text=candidate.question_b,
        ),
        relation=candidate.relation,
        confidence=candidate.confidence,
        differences=[],
        reason="live price check",
    )


async def poll_once(
    pairs_path: str,
    min_arb_edge: float = 0.02,
    min_confidence: float = 0.70,
) -> list[dict]:
    path = Path(pairs_path)
    if not path.exists():
        print(f"[poller] arb_pairs.json not found at {pairs_path}. Run pipeline first.")
        return []

    with open(path, encoding="utf-8") as f:
        data = ArbPairsOutput.model_validate(json.load(f))

    if not data.pairs:
        print("[poller] No candidates in arb_pairs.json.")
        return []

    print(f"[poller] Fetching live prices for {len(data.pairs)} candidate pairs...")
    found: list[dict] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Fetch all prices concurrently
        tasks_a = [fetch_live_price(c.exchange_a, c.native_id_a, client) for c in data.pairs]
        tasks_b = [fetch_live_price(c.exchange_b, c.native_id_b, client) for c in data.pairs]
        prices_a = await asyncio.gather(*tasks_a)
        prices_b = await asyncio.gather(*tasks_b)

    for candidate, (ya, na), (yb, nb) in zip(data.pairs, prices_a, prices_b):
        if ya is None or yb is None:
            print(f"[poller] SKIP {candidate.uid_a} x {candidate.uid_b} — price unavailable")
            continue

        pair = _candidate_to_pair(candidate, ya, na, yb, nb)
        opp = score_pair(pair, min_confidence=min_confidence, min_arb_edge=min_arb_edge)

        if opp:
            record = {
                "edge": opp.arb_edge,
                "relation": candidate.relation.value,
                "confidence": candidate.confidence,
                "strategy": opp.strategy,
                "uid_a": candidate.uid_a,
                "uid_b": candidate.uid_b,
                "question_a": candidate.question_a,
                "question_b": candidate.question_b,
                "yes_price_a": ya,
                "yes_price_b": yb,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            found.append(record)
            print(
                f"\n  *** OPPORTUNITY ***"
                f"\n  Edge:     {opp.arb_edge:.4f}"
                f"\n  Relation: {candidate.relation.value}  (conf={candidate.confidence:.2f})"
                f"\n  A [{candidate.exchange_a}] {candidate.question_a[:70]}"
                f"\n  B [{candidate.exchange_b}] {candidate.question_b[:70]}"
                f"\n  Strategy: {opp.strategy}"
            )

    if not found:
        print(f"[poller] No opportunities above edge={min_arb_edge} at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    return found


async def poll_loop(
    pairs_path: str,
    interval: int,
    min_arb_edge: float,
    min_confidence: float,
) -> None:
    print(f"[poller] Starting poll loop (interval={interval}s, min_edge={min_arb_edge})")
    while True:
        await poll_once(pairs_path, min_arb_edge=min_arb_edge, min_confidence=min_confidence)
        print(f"[poller] Sleeping {interval}s...")
        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live arb price poller")
    parser.add_argument(
        "--pairs",
        default="SemanticMatchPipeline/arb_pairs.json",
        help="Path to arb_pairs.json written by pipeline.py",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Poll every N seconds. 0 = poll once and exit.",
    )
    parser.add_argument(
        "--min-arb-edge",
        type=float,
        default=0.02,
        help="Minimum arb edge to alert on",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="Minimum LLM confidence to consider a pair",
    )
    args = parser.parse_args()

    if args.interval > 0:
        asyncio.run(poll_loop(args.pairs, args.interval, args.min_arb_edge, args.min_confidence))
    else:
        asyncio.run(poll_once(args.pairs, args.min_arb_edge, args.min_confidence))


if __name__ == "__main__":
    main()
