"""
Scout Pipeline: one-command fetch → index → discover pairs.

Fetches live markets from the requested exchanges, indexes them into an
isolated vector DB, then runs the Scout in both directions to find
cross-exchange arbitrage pairs.  Results land in a verified_pairs file
ready to feed into historical_data.py and backtest.py.

Usage (test DB — production state untouched):
    python scout_pipeline.py --test-db --limit 500

Promote to production when satisfied:
    cp verified_pairs.test.json verified_pairs.json

Then backtest:
    python historical_data.py --pairs verified_pairs.json --start 2025-01-01 --end 2025-04-29
    python backtest.py --pairs verified_pairs.json --start 2025-01-01 --end 2025-04-29
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Allow running as a script or via `python -m scout_pipeline`
sys.path.insert(0, str(Path(__file__).parent))

from models import Exchange, UnifiedMarket
from pipeline import run_pipeline, setup_logging
from vector_store import setup_client, upsert_markets
from scout import Scout, DEFAULT_THRESHOLD, DEFAULT_LLM_THRESHOLD, DEFAULT_ACCEPT_LABELS, STORAGE_FILE

logger = logging.getLogger("scout_pipeline")

# Test-DB defaults (separate from production state)
TEST_QDRANT_PATH = "qdrant_test"
TEST_COLLECTION = "markets_test"
TEST_STORAGE_FILE = "verified_pairs.test.json"


# ─── Fetch + Index ────────────────────────────────────────────────────────────

async def fetch_and_index(
    exchanges: list[Exchange],
    limit: int,
    qdrant_path: str,
    collection: str,
    status: str = "open",
    category_include: Optional[list[str]] = None,
    kalshi_series: Optional[list[str]] = None,
    batch_size: int = 64,
) -> int:
    """Fetch live markets, optionally filter by category, and index into Qdrant.

    Returns the number of markets indexed.
    """
    logger.info(f"Fetching up to {limit} markets per exchange from {[e.value for e in exchanges]}...")
    markets: list[UnifiedMarket] = await run_pipeline(exchanges, status, limit, kalshi_series=kalshi_series)
    logger.info(f"Fetched {len(markets)} total markets.")

    # Optional category filter (case-insensitive substring on category/tags/event_title)
    if category_include:
        needles = [c.lower() for c in category_include]
        filtered = []
        for m in markets:
            haystack = " ".join(filter(None, [
                (m.category or "").lower(),
                " ".join(m.tags or []).lower(),
                (m.event_title or "").lower(),
                m.question.lower(),
            ]))
            if any(n in haystack for n in needles):
                filtered.append(m)
        logger.info(
            f"Category filter '{','.join(category_include)}': {len(filtered)}/{len(markets)} markets kept."
        )
        markets = filtered

    if not markets:
        logger.warning("No markets to index after filtering.")
        return 0

    data = [m.to_dict() for m in markets]
    client = setup_client(path=qdrant_path, collection=collection)

    # Wipe the collection so stale markets from a previous run don't pollute results
    try:
        client.delete_collection(collection)
        logger.info(f"Cleared existing collection '{collection}'.")
    except Exception:
        pass  # Collection didn't exist yet

    logger.info(f"Indexing {len(data)} markets into '{qdrant_path}' / collection '{collection}'...")
    await asyncio.to_thread(
        upsert_markets,
        data,
        client,
        batch_size,
        0,  # parallel=0 → all cores
        collection,
    )
    logger.info("Indexing complete.")
    return len(data)


# ─── Scout Both Directions ────────────────────────────────────────────────────

def run_scout(
    exchanges: list[Exchange],
    qdrant_path: str,
    collection: str,
    storage_file: str,
    threshold: float,
    llm_threshold: float,
    candidates: int,
    min_confidence: float,
    accept_labels: set[str],
    limit: int,
    dry_run: bool,
    llm_call_delay: float = 0.0,
):
    """Run Scout for every exchange pair (both directions) and persist results."""
    scout = Scout(
        dry_run=dry_run,
        similarity_threshold=threshold,
        llm_threshold=llm_threshold,
        qdrant_path=qdrant_path,
        collection=collection,
        storage_file=storage_file,
        accept_labels=accept_labels,
        min_confidence=min_confidence,
        llm_call_delay=llm_call_delay,
    )

    for source in exchanges:
        logger.info(f"=== Scouting {source.value} → other exchanges ===")
        scout.find_matches(source, limit=limit, candidates_per_source=candidates)

    count = len(scout.verified_pairs)
    logger.info(f"Discovery complete. {count} verified pair(s) in '{storage_file}'.")
    return count


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Fetch live markets, index into a vector DB, and discover cross-exchange pairs."
    )

    # Test DB shortcut
    parser.add_argument(
        "--test-db", action="store_true",
        help=(
            f"Use isolated test DB (path={TEST_QDRANT_PATH}, "
            f"collection={TEST_COLLECTION}, "
            f"output={TEST_STORAGE_FILE}). "
            "Production qdrant_data/ and verified_pairs.json are untouched."
        ),
    )

    # Explicit overrides
    parser.add_argument("--qdrant-path", type=str, default=None,
                        help="Qdrant storage path (overrides --test-db default)")
    parser.add_argument("--collection", type=str, default=None,
                        help="Qdrant collection name (overrides --test-db default)")
    parser.add_argument("--storage-file", type=str, default=None,
                        help="Output pairs JSON file (overrides --test-db default)")

    # Fetch options
    parser.add_argument(
        "--exchanges", nargs="+",
        choices=[e.value for e in Exchange],
        default=["kalshi", "polymarket"],
        help="Exchanges to fetch from (default: kalshi polymarket)",
    )
    parser.add_argument("--limit", type=int, default=500,
                        help="Max markets per exchange to fetch and scan (default: 500)")
    parser.add_argument("--status", type=str, default="open",
                        choices=["open", "closed", "settled"],
                        help="Market status filter (default: open)")
    parser.add_argument(
        "--category-include", type=str, default=None,
        help="Comma-separated substrings to filter markets by category/tags/question before indexing "
             "(e.g. 'politics,sports,crypto'). If omitted all markets are indexed.",
    )
    parser.add_argument(
        "--kalshi-series", type=str, default=None,
        help="Comma-separated Kalshi series tickers to fetch instead of the default paginated list "
             "(e.g. 'RATECUT,BTC,KXCPICOMBO'). Bypasses sports-heavy default pagination.",
    )

    # Scout tuning
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Vector similarity threshold for candidate retrieval (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--llm-threshold", type=float, default=DEFAULT_LLM_THRESHOLD,
                        help=f"Minimum similarity score to escalate to LLM; below this UNRELATED/AMBIGUOUS are assumed (default: {DEFAULT_LLM_THRESHOLD})")
    parser.add_argument("--candidates", type=int, default=10,
                        help="Max candidates per source market (default: 10)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum LLM confidence to accept a pair (default: 0.7)")
    parser.add_argument(
        "--accept-labels", type=str, default=None,
        help="Comma-separated labels to accept (default: IDENTICAL,COMPLEMENT)",
    )

    parser.add_argument("--llm-call-delay", type=float, default=0.0,
                        help="Seconds to sleep between LLM calls to avoid rate limits (default: 0.0)")

    # Flow control
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching and indexing; use whatever is already in the DB")
    parser.add_argument("--skip-scout", action="store_true",
                        help="Only fetch+index, skip pair discovery")
    parser.add_argument("--dry-run", action="store_true",
                        help="Index markets but skip LLM calls during scouting")

    args = parser.parse_args()

    # Resolve DB / file paths (--test-db sets defaults; explicit flags override)
    if args.test_db:
        qdrant_path = args.qdrant_path or TEST_QDRANT_PATH
        collection = args.collection or TEST_COLLECTION
        storage_file = args.storage_file or TEST_STORAGE_FILE
    else:
        qdrant_path = args.qdrant_path or "qdrant_data"
        collection = args.collection or "markets"
        storage_file = args.storage_file or STORAGE_FILE

    exchanges = [Exchange(e) for e in args.exchanges]
    accept_labels = (
        {label.strip().upper() for label in args.accept_labels.split(",") if label.strip()}
        if args.accept_labels else DEFAULT_ACCEPT_LABELS
    )
    category_include = (
        [c.strip() for c in args.category_include.split(",")]
        if args.category_include else None
    )
    kalshi_series = (
        [s.strip().upper() for s in args.kalshi_series.split(",") if s.strip()]
        if args.kalshi_series else None
    )

    logger.info("=" * 60)
    logger.info(f"DB path   : {qdrant_path}")
    logger.info(f"Collection: {collection}")
    logger.info(f"Output    : {storage_file}")
    logger.info(f"Exchanges : {[e.value for e in exchanges]}")
    logger.info("=" * 60)

    # Step 1: Fetch + Index
    if not args.skip_fetch:
        indexed = asyncio.run(fetch_and_index(
            exchanges=exchanges,
            limit=args.limit,
            qdrant_path=qdrant_path,
            collection=collection,
            status=args.status,
            category_include=category_include,
            kalshi_series=kalshi_series,
        ))
        if indexed == 0 and not args.skip_scout:
            logger.error("No markets indexed — nothing to scout. Exiting.")
            sys.exit(1)
    else:
        logger.info("--skip-fetch: using existing DB contents.")

    # Step 2: Scout
    if not args.skip_scout:
        count = run_scout(
            exchanges=exchanges,
            qdrant_path=qdrant_path,
            collection=collection,
            storage_file=storage_file,
            threshold=args.threshold,
            llm_threshold=args.llm_threshold,
            candidates=args.candidates,
            min_confidence=args.min_confidence,
            accept_labels=accept_labels,
            limit=args.limit,
            dry_run=args.dry_run,
            llm_call_delay=args.llm_call_delay,
        )
        if count == 0:
            logger.warning(
                "No pairs found. Try lowering --threshold, increasing --limit, "
                "or removing --category-include to broaden the corpus."
            )
        else:
            logger.info(
                f"\nNext steps:\n"
                f"  1. Inspect: python -c \"import json; print(json.load(open('{storage_file}')))\"\n"
                f"  2. Promote: cp {storage_file} verified_pairs.json\n"
                f"  3. Backfill: python historical_data.py --pairs verified_pairs.json "
                f"--start 2025-01-01 --end 2025-04-29\n"
                f"  4. Backtest: python backtest.py --pairs verified_pairs.json "
                f"--start 2025-01-01 --end 2025-04-29"
            )
    else:
        logger.info("--skip-scout: skipping pair discovery.")


if __name__ == "__main__":
    main()
