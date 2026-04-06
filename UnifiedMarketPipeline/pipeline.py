"""
Pipeline orchestrator: fetch from all exchanges, normalize, and export.

Usage:
    python pipeline.py                    # Fetch all exchanges, export JSON
    python pipeline.py --exchanges kalshi polymarket
    python pipeline.py --output markets.jsonl --format jsonl
    python pipeline.py --limit 50         # Limit per exchange
"""

from __future__ import annotations
import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from models import UnifiedMarket, Exchange
from adapters import (
    KalshiAdapter,
    PolymarketAdapter,
    ManifoldAdapter,
    MetaculusAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


ADAPTER_MAP = {
    Exchange.KALSHI: KalshiAdapter,
    Exchange.POLYMARKET: PolymarketAdapter,
    Exchange.MANIFOLD: ManifoldAdapter,
    Exchange.METACULUS: MetaculusAdapter,
}


async def fetch_exchange(
    exchange: Exchange,
    client: httpx.AsyncClient,
    status: str = "open",
    limit: int = 200,
) -> list[UnifiedMarket]:
    """Fetch and normalize all markets from one exchange."""
    adapter_cls = ADAPTER_MAP[exchange]
    adapter = adapter_cls(client=client)
    markets = []
    count = 0

    try:
        async for market in adapter.fetch_markets(status=status, limit=limit):
            markets.append(market)
            count += 1
            if count >= limit:
                break
    except httpx.HTTPStatusError as e:
        logger.error(f"[{exchange.value}] HTTP {e.response.status_code}: {e}")
    except Exception as e:
        logger.error(f"[{exchange.value}] Error: {e}")

    logger.info(f"[{exchange.value}] Fetched {len(markets)} markets")
    return markets


async def run_pipeline(
    exchanges: list[Exchange],
    status: str = "open",
    limit: int = 200,
) -> list[UnifiedMarket]:
    """Fetch from all requested exchanges concurrently."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            fetch_exchange(exchange, client, status, limit)
            for exchange in exchanges
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_markets = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Pipeline error: {result}")
        else:
            all_markets.extend(result)

    logger.info(f"Total: {len(all_markets)} markets across {len(exchanges)} exchanges")
    return all_markets


def export_json(markets: list[UnifiedMarket], path: Path):
    """Export as a single JSON array."""
    data = [m.to_dict() for m in markets]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Exported {len(markets)} markets to {path}")


def export_jsonl(markets: list[UnifiedMarket], path: Path):
    """Export as newline-delimited JSON (one market per line)."""
    with open(path, "w") as f:
        for m in markets:
            f.write(json.dumps(m.to_dict(), default=str) + "\n")
    logger.info(f"Exported {len(markets)} markets to {path}")


def export_embedding_corpus(markets: list[UnifiedMarket], path: Path):
    """
    Export just the embedding text + metadata for each market.
    This is the file you feed to your embedding model.
    """
    records = []
    for m in markets:
        records.append({
            "uid": m.uid,
            "exchange": m.exchange.value,
            "text": m.embedding_text,
            "content_hash": m.content_hash,
            "yes_price": m.yes_price,
            "volume_24h": m.volume_24h,
            "close_at": m.close_at.isoformat() if m.close_at else None,
            "status": m.status.value,
        })
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info(f"Exported embedding corpus ({len(records)} docs) to {path}")


def print_summary(markets: list[UnifiedMarket]):
    """Print a summary table to stdout."""
    by_exchange = {}
    for m in markets:
        by_exchange.setdefault(m.exchange.value, []).append(m)

    print("\n" + "=" * 70)
    print(f"  PREDICTION MARKET PIPELINE — {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    for ex, ms in sorted(by_exchange.items()):
        print(f"\n  {ex.upper()} ({len(ms)} markets)")
        print(f"  {'─' * 60}")
        for m in ms[:5]:
            price_str = f"${m.yes_price:.2f}" if m.yes_price else "N/A"
            q = m.question[:55] + "…" if len(m.question) > 55 else m.question
            print(f"    {price_str:>6}  {q}")
        if len(ms) > 5:
            print(f"    ... and {len(ms) - 5} more")
    print(f"\n  Total: {len(markets)} markets")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified prediction market data pipeline"
    )
    parser.add_argument(
        "--exchanges", nargs="+",
        choices=[e.value for e in Exchange],
        default=[e.value for e in Exchange],
        help="Exchanges to fetch from",
    )
    parser.add_argument(
        "--status", default="open",
        choices=["open", "closed", "settled"],
        help="Market status filter",
    )
    parser.add_argument("--limit", type=int, default=200, help="Max markets per exchange")
    parser.add_argument("--output", type=str, default="markets.json", help="Output file path")
    parser.add_argument(
        "--format", choices=["json", "jsonl", "embedding"], default="json",
        help="Output format",
    )
    parser.add_argument("--summary", action="store_true", help="Print summary to stdout")

    args = parser.parse_args()
    exchanges = [Exchange(e) for e in args.exchanges]

    markets = asyncio.run(run_pipeline(exchanges, args.status, args.limit))

    path = Path(args.output)
    if args.format == "json":
        export_json(markets, path)
    elif args.format == "jsonl":
        export_jsonl(markets, path)
    elif args.format == "embedding":
        export_embedding_corpus(markets, path)

    if args.summary or True:  # Always print summary
        print_summary(markets)


if __name__ == "__main__":
    main()
