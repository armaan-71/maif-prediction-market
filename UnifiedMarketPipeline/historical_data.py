"""
Historical price data: fetchers + cache + manager.

Pulls per-market price time series from:
  - Polymarket: native CLOB /prices-history (no auth)
  - Kalshi:     Oddpool /historical/kalshi/top-of-book (X-API-Key)

Caches results as JSONL on disk so repeated backtests don't re-download.
Manifold and Metaculus are not supported by either source — markets on
those exchanges are skipped (the backtest engine will exclude pairs that
involve them).

Standalone CLI:
    python historical_data.py --pairs verified_pairs.json \
        --start 2025-01-01 --end 2025-12-31 [--fidelity 60]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import httpx
from dotenv import load_dotenv

from models import UnifiedMarket, Exchange, MarketStatus

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / "data" / "historical"
SUPPORTED_EXCHANGES = {Exchange.KALSHI, Exchange.POLYMARKET}


# ─── Tick model ───────────────────────────────────────────────────────────────

@dataclass
class PriceTick:
    timestamp: datetime
    yes_price: float
    no_price: Optional[float] = None
    volume: Optional[float] = None

    def to_jsonl(self) -> str:
        return json.dumps({
            "t": self.timestamp.replace(tzinfo=timezone.utc).timestamp()
                 if self.timestamp.tzinfo is None
                 else self.timestamp.timestamp(),
            "y": self.yes_price,
            "n": self.no_price,
            "v": self.volume,
        })

    @classmethod
    def from_jsonl(cls, line: str) -> "PriceTick":
        d = json.loads(line)
        return cls(
            timestamp=datetime.fromtimestamp(d["t"], tz=timezone.utc),
            yes_price=d["y"],
            no_price=d.get("n"),
            volume=d.get("v"),
        )


# ─── Cache ────────────────────────────────────────────────────────────────────

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    **kwargs,
) -> Optional[httpx.Response]:
    """HTTP request with exponential backoff on 429 / 5xx. Honors Retry-After.

    Returns the successful Response, or None if all retries are exhausted or a
    non-retryable client error (4xx other than 429) is encountered.
    """
    for attempt in range(max_retries + 1):
        try:
            r = client.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            if attempt == max_retries:
                logger.error(f"Network error after {max_retries} retries: {exc}")
                return None
            delay = min(60.0, base_delay * (2 ** attempt))
            logger.warning(f"Network error ({exc}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)
            continue

        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries:
                logger.error(f"HTTP {r.status_code} after {max_retries} retries: {url}")
                return None
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = min(60.0, base_delay * (2 ** attempt))
            else:
                delay = min(60.0, base_delay * (2 ** attempt))
            logger.warning(
                f"HTTP {r.status_code} ({url}), retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
            continue

        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error (non-retryable): {exc}")
            return None

        return r

    return None


class HistoricalDataCache:
    """JSONL-per-market cache. Files are append-only, sorted ascending by ts."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache of the latest timestamp per market to avoid re-reading on every append.
        self._latest: dict[str, Optional[datetime]] = {}

    def _path(self, exchange: str, market_id: str) -> Path:
        safe = _SAFE_ID.sub("_", market_id)[:200]
        d = self.cache_dir / exchange
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe}.jsonl"

    def _cache_key(self, exchange: str, market_id: str) -> str:
        return f"{exchange}:{market_id}"

    def load(self, exchange: str, market_id: str) -> list[PriceTick]:
        path = self._path(exchange, market_id)
        if not path.exists():
            return []
        ticks: list[PriceTick] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    ticks.append(PriceTick.from_jsonl(line))
        ticks.sort(key=lambda t: t.timestamp)
        return ticks

    def append(self, exchange: str, market_id: str, ticks: list[PriceTick]) -> None:
        if not ticks:
            return
        path = self._path(exchange, market_id)
        latest = self.latest_timestamp(exchange, market_id)
        written = 0
        with path.open("a") as f:
            for t in sorted(ticks, key=lambda x: x.timestamp):
                if latest is not None and t.timestamp <= latest:
                    continue
                f.write(t.to_jsonl() + "\n")
                written += 1
                if latest is None or t.timestamp > latest:
                    latest = t.timestamp
        if written > 0:
            # Update in-memory cache
            self._latest[self._cache_key(exchange, market_id)] = latest

    def latest_timestamp(self, exchange: str, market_id: str) -> Optional[datetime]:
        key = self._cache_key(exchange, market_id)
        if key not in self._latest:
            ticks = self.load(exchange, market_id)
            self._latest[key] = ticks[-1].timestamp if ticks else None
        return self._latest[key]

    def in_range(
        self, exchange: str, market_id: str, start: datetime, end: datetime
    ) -> list[PriceTick]:
        return [t for t in self.load(exchange, market_id) if start <= t.timestamp <= end]


# ─── Fetchers ─────────────────────────────────────────────────────────────────

class PolymarketHistoricalFetcher:
    BASE = "https://clob.polymarket.com"

    def __init__(self, client: Optional[httpx.Client] = None):
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None

    def close(self):
        if self._owns_client:
            self._client.close()

    def fetch(
        self,
        token_id: str,
        start: datetime,
        end: datetime,
        fidelity_minutes: int = 60,
    ) -> list[PriceTick]:
        params = {
            "market": token_id,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "fidelity": int(fidelity_minutes),
        }
        r = _request_with_retry(
            self._client, "GET", f"{self.BASE}/prices-history", params=params
        )
        if r is None:
            logger.error(f"[polymarket] {token_id}: all retries failed, returning empty")
            return []

        history = r.json().get("history", [])
        ticks = [
            PriceTick(
                timestamp=datetime.fromtimestamp(point["t"], tz=timezone.utc),
                yes_price=float(point["p"]),
            )
            for point in history
        ]
        logger.info(f"[polymarket] {token_id}: fetched {len(ticks)} ticks")
        return ticks


class OddpoolKalshiFetcher:
    BASE = "https://api.oddpool.com"

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.Client] = None):
        self.api_key = api_key or os.getenv("ODDPOOL_API_KEY")
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None

    def close(self):
        if self._owns_client:
            self._client.close()

    def fetch(
        self,
        market_ticker: str,
        start: datetime,
        end: datetime,
        granularity: str = "1h",
    ) -> list[PriceTick]:
        if not self.api_key:
            logger.warning(
                f"[oddpool/kalshi] {market_ticker}: ODDPOOL_API_KEY not set; skipping"
            )
            return []

        # Oddpool paginates via start_ts; loop until we cover [start, end]
        # or the API returns nothing new.
        ticks: list[PriceTick] = []
        cursor = start
        seen_ts: set[float] = set()
        page_limit = 200

        while cursor < end:
            params = {
                "market_id": market_ticker,
                "granularity": granularity,
                "start_ts": int(cursor.timestamp()),
                "end_ts": int(end.timestamp()),
                "limit": page_limit,
            }
            r = _request_with_retry(
                self._client,
                "GET",
                f"{self.BASE}/historical/kalshi/top-of-book",
                params=params,
                headers={"X-API-Key": self.api_key},
            )
            if r is None:
                logger.error(f"[oddpool/kalshi] {market_ticker}: all retries failed, stopping pagination")
                break

            data = r.json()
            rows = data.get("data") or data.get("snapshots") or data.get("results") or []
            if not rows:
                break

            new_count = 0
            last_ts = cursor
            for row in rows:
                ts_raw = row.get("timestamp") or row.get("t") or row.get("ts")
                if ts_raw is None:
                    continue
                if isinstance(ts_raw, (int, float)):
                    if ts_raw > 1e12:  # milliseconds
                        ts_raw /= 1000
                    ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                else:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                if ts.timestamp() in seen_ts:
                    continue
                seen_ts.add(ts.timestamp())

                mid = row.get("mid")
                bid = row.get("best_yes_bid")
                ask = row.get("best_yes_ask")
                if mid is None and bid is not None and ask is not None:
                    mid = (bid + ask) / 2
                if mid is None:
                    continue

                yes_price = float(mid)
                if yes_price > 1.5:
                    yes_price /= 100.0  # Kalshi cents
                ticks.append(PriceTick(
                    timestamp=ts,
                    yes_price=yes_price,
                    no_price=1.0 - yes_price,
                    volume=row.get("volume"),
                ))
                new_count += 1
                if ts > last_ts:
                    last_ts = ts

            if new_count == 0 or last_ts <= cursor:
                break
            cursor = last_ts
            if len(rows) < page_limit:
                break
            time.sleep(0.2)  # proactive rate-limit courtesy between pages

        ticks.sort(key=lambda t: t.timestamp)
        logger.info(f"[oddpool/kalshi] {market_ticker}: fetched {len(ticks)} ticks")
        return ticks


# ─── Manager ──────────────────────────────────────────────────────────────────

def _polymarket_yes_token(market: UnifiedMarket) -> Optional[str]:
    """Pick the YES outcome's CLOB token id."""
    for o in market.outcomes:
        if o.token_id and o.label.strip().lower() in ("yes", "y", "true"):
            return o.token_id
    if market.outcomes and market.outcomes[0].token_id:
        return market.outcomes[0].token_id
    return None


class HistoricalDataManager:
    def __init__(
        self,
        cache: Optional[HistoricalDataCache] = None,
        oddpool_api_key: Optional[str] = None,
        polymarket_fetcher: Optional[PolymarketHistoricalFetcher] = None,
        kalshi_fetcher: Optional[OddpoolKalshiFetcher] = None,
    ):
        self.cache = cache or HistoricalDataCache()
        self.polymarket = polymarket_fetcher or PolymarketHistoricalFetcher()
        self.kalshi = kalshi_fetcher or OddpoolKalshiFetcher(api_key=oddpool_api_key)

    def close(self):
        self.polymarket.close()
        self.kalshi.close()

    def get_history(
        self,
        market: UnifiedMarket,
        start: datetime,
        end: datetime,
        fidelity_minutes: int = 60,
        refresh: bool = True,
    ) -> list[PriceTick]:
        """Cache-first read; fetches only ticks newer than what's cached."""
        if market.exchange not in SUPPORTED_EXCHANGES:
            logger.debug(
                f"[{market.uid}] exchange {market.exchange.value} not supported, skipping"
            )
            return []

        ex = market.exchange.value
        cached = self.cache.load(ex, market.native_id)

        # Resolved markets: cache is immutable.
        already_resolved = market.status == MarketStatus.SETTLED and cached
        latest = cached[-1].timestamp if cached else None

        if refresh and not already_resolved and (latest is None or latest < end):
            fetch_start = max(start, latest) if latest else start
            new_ticks = self._fetch(market, fetch_start, end, fidelity_minutes)
            if new_ticks:
                self.cache.append(ex, market.native_id, new_ticks)

        return self.cache.in_range(ex, market.native_id, start, end)

    def _fetch(
        self,
        market: UnifiedMarket,
        start: datetime,
        end: datetime,
        fidelity_minutes: int,
    ) -> list[PriceTick]:
        if market.exchange == Exchange.POLYMARKET:
            token = _polymarket_yes_token(market)
            if not token:
                logger.warning(f"[{market.uid}] no Polymarket YES token id, skipping")
                return []
            return self.polymarket.fetch(token, start, end, fidelity_minutes)

        if market.exchange == Exchange.KALSHI:
            granularity = "5m" if fidelity_minutes >= 5 else "1m"
            return self.kalshi.fetch(market.native_id, start, end, granularity)

        return []

    def materialize_snapshots(
        self,
        market: UnifiedMarket,
        start: datetime,
        end: datetime,
        fidelity_minutes: int = 60,
        refresh: bool = True,
    ) -> list[UnifiedMarket]:
        """
        Replay the price ticks as a sequence of UnifiedMarket snapshots, each
        with .yes_price / .no_price / .fetched_at set from a tick. The final
        snapshot inherits the source market's resolution fields if it settled.
        """
        ticks = self.get_history(market, start, end, fidelity_minutes, refresh=refresh)
        if not ticks:
            return []

        snapshots: list[UnifiedMarket] = []
        for i, tick in enumerate(ticks):
            snap = market.model_copy(deep=True)
            snap.yes_price = tick.yes_price
            snap.no_price = (
                tick.no_price if tick.no_price is not None else 1.0 - tick.yes_price
            )
            snap.fetched_at = tick.timestamp
            # Only the last snapshot reflects resolution.
            if i < len(ticks) - 1:
                snap.status = MarketStatus.OPEN
                snap.result = None
                snap.resolved_at = None
            snapshots.append(snap)
        return snapshots


# ─── Helpers for batch download ───────────────────────────────────────────────

def load_pairs(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return json.load(f)


def markets_from_pairs(pairs: list[dict]) -> list[UnifiedMarket]:
    """
    Construct UnifiedMarket objects for each unique market in verified_pairs.json.

    Newer pairs files (written by the fixed Scout) include outcomes with token_id,
    which is required for the Polymarket historical fetcher. Older files that only
    carry uid/exchange/question/yes_price are still handled as minimal stubs.
    """
    from models import Outcome as _Outcome, MarketStatus as _MS  # avoid circular at module load

    seen: dict[str, UnifiedMarket] = {}
    for p in pairs:
        for side in ("market_a", "market_b"):
            m = p[side]
            if m["uid"] in seen:
                continue
            try:
                exchange = Exchange(m["exchange"])
            except ValueError:
                continue
            native_id = m.get("native_id") or (
                m["uid"].split(":", 1)[1] if ":" in m["uid"] else m["uid"]
            )

            # Hydrate outcomes including token_id when present (new-format pairs)
            raw_outcomes = m.get("outcomes", [])
            outcomes = []
            for o in raw_outcomes:
                try:
                    outcomes.append(_Outcome(**o))
                except Exception:
                    pass

            # Carry status/result forward so the settled-market cache optimization works
            raw_status = m.get("status", "unknown")
            try:
                status = _MS(raw_status)
            except ValueError:
                status = _MS.UNKNOWN

            seen[m["uid"]] = UnifiedMarket(
                exchange=exchange,
                native_id=native_id,
                question=m.get("question", ""),
                yes_price=m.get("yes_price"),
                no_price=m.get("no_price"),
                url=m.get("url"),
                outcomes=outcomes,
                status=status,
                result=m.get("result"),
            )
    return list(seen.values())


def _parse_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Pre-fetch historical price data.")
    parser.add_argument("--pairs", required=True, help="verified_pairs.json")
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2025-01-01")
    parser.add_argument("--end", required=True, help="ISO date, e.g. 2025-12-31")
    parser.add_argument("--fidelity", type=int, default=60,
                        help="Granularity in minutes (default 60)")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pairs))
    markets = markets_from_pairs(pairs)
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    mgr = HistoricalDataManager()
    try:
        for m in markets:
            if m.exchange not in SUPPORTED_EXCHANGES:
                logger.info(f"skip {m.uid} (exchange {m.exchange.value} unsupported)")
                continue
            ticks = mgr.get_history(m, start, end, args.fidelity)
            print(f"{m.uid}: {len(ticks)} ticks cached")
    finally:
        mgr.close()


if __name__ == "__main__":
    main()
