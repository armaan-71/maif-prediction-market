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

import vector_store
from models import UnifiedMarket, Exchange
from adapters import (
    KalshiAdapter,
    PolymarketAdapter,
    ManifoldAdapter,
    MetaculusAdapter,
)

logger = logging.getLogger("pipeline")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Call from __main__ blocks only."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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
    series_tickers: Optional[list[str]] = None,
) -> list[UnifiedMarket]:
    """Fetch and normalize all markets from one exchange."""
    adapter_cls = ADAPTER_MAP[exchange]
    adapter = adapter_cls(client=client)
    markets = []
    count = 0

    kwargs: dict = {"status": status, "limit": limit}
    if series_tickers:
        kwargs["series_tickers"] = series_tickers

    try:
        async for market in adapter.fetch_markets(**kwargs):
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
    kalshi_series: Optional[list[str]] = None,
) -> list[UnifiedMarket]:
    """Fetch from all requested exchanges concurrently."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            fetch_exchange(exchange, client, status, limit,
                           series_tickers=kalshi_series if exchange == Exchange.KALSHI else None)
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


async def export_to_vector_store(
    markets: list[UnifiedMarket],
    batch_size: int = 64,
    parallel: int = 0,
):
    """Index markets in the Qdrant vector database."""
    if not markets:
        return

    logger.info(f"Indexing {len(markets)} markets into Qdrant...")
    data = [m.to_dict() for m in markets]

    # Run in thread pool because embedding generation is CPU intensive
    await asyncio.to_thread(
        vector_store.upsert_markets, data, None, batch_size, parallel
    )


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


async def async_main():
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
        "--format", choices=["json", "jsonl", "embedding", "none"], default="json",
        help="Output format (use 'none' if only using --vector)",
    )
    parser.add_argument("--vector", action="store_true", help="Upsert to Qdrant vector store")
    parser.add_argument(
        "--vector-batch-size", type=int, default=64,
        help="Docs per embed+upsert batch (default: 64)",
    )
    parser.add_argument(
        "--vector-parallel", type=int, default=0,
        help="Embedding workers (0=all cores, 1=serial). Default: 0",
    )
    parser.add_argument("--summary", action="store_true", help="Print summary to stdout")

    args = parser.parse_args()
    exchanges = [Exchange(e) for e in args.exchanges]

    markets = await run_pipeline(exchanges, args.status, args.limit)

    if args.vector:
        await export_to_vector_store(
            markets, args.vector_batch_size, args.vector_parallel
        )

    path = Path(args.output)
    if args.format == "json":
        export_json(markets, path)
    elif args.format == "jsonl":
        export_jsonl(markets, path)
    elif args.format == "embedding":
        export_embedding_corpus(markets, path)

    if args.summary:
        print_summary(markets)


def main():
    setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

