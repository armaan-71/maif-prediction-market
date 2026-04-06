"""
Exchange adapters: normalize raw API responses into UnifiedMarket objects.

Each adapter handles one exchange's quirks—auth, pagination, field mapping—
and emits a stream of UnifiedMarket instances.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import httpx

from models import (
    Exchange, MarketStatus, OutcomeType, ResolutionSource,
    Outcome, UnifiedMarket,
)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', level=logging.DEBUG)


# ─── Base Adapter ─────────────────────────────────────────────────────────────

class ExchangeAdapter(ABC):
    """
    Abstract base for all exchange adapters.

    Subclasses implement `fetch_markets()` which yields UnifiedMarket objects.
    The base class provides shared HTTP helpers and rate-limit handling.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self.client = client or httpx.AsyncClient(timeout=30.0)

    @abstractmethod
    async def fetch_markets(
        self,
        status: Optional[str] = "open",
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[UnifiedMarket]:
        """Yield normalized markets from this exchange."""
        ...

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def _parse_ts(val) -> Optional[datetime]:
        """Parse a timestamp from ISO string, unix seconds, or unix ms."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            # Heuristic: if > 1e12, it's milliseconds
            if val > 1e12:
                val = val / 1000
            return datetime.fromtimestamp(val, tz=timezone.utc)
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None


# ─── Kalshi Adapter ───────────────────────────────────────────────────────────

class KalshiAdapter(ExchangeAdapter):
    """
    Kalshi REST API v2.

    Base URL: https://api.elections.kalshi.com/trade-api/v2
    Public endpoints (no auth needed): GET /markets, GET /events
    Hierarchy: Series → Event → Market (binary contract)
    Pagination: cursor-based
    """

    BASE = "https://api.elections.kalshi.com/trade-api/v2"

    STATUS_MAP = {
        "active": MarketStatus.OPEN,
        "open": MarketStatus.OPEN,
        "initialized": MarketStatus.OPEN,
        "closed": MarketStatus.CLOSED,
        "settled": MarketStatus.SETTLED,
        "determined": MarketStatus.SETTLED,
    }

    async def fetch_markets(
        self,
        status: Optional[str] = "open",
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[UnifiedMarket]:
        params = {
            "limit": min(limit, 1000),
            # Exclude multivariate event combos (parlays).  These bundle
            # multiple legs into one contract and have no standalone
            # semantic question — useless for cross-exchange matching.
            "mve_filter": "exclude",
        }
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        while True:
            resp = await self.client.get(f"{self.BASE}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()

            for m in data.get("markets", []):
                # Defensive: skip any MVE that slipped through the filter
                if self._is_mve(m):
                    continue
                yield self._normalize(m)

            next_cursor = data.get("cursor")
            if not next_cursor or len(data.get("markets", [])) == 0:
                break
            params["cursor"] = next_cursor

    @staticmethod
    def _is_mve(m: dict) -> bool:
        """
        Detect multivariate event (parlay/combo) markets.

        MVEs have:
          - mve_collection_ticker set
          - mve_selected_legs array with multiple entries
          - ticker containing 'MVE' or 'MULTIGAME'
          - yes_sub_title with comma-separated leg descriptions
        """
        if m.get("mve_collection_ticker"):
            return True
        if m.get("mve_selected_legs"):
            return True
        ticker = m.get("ticker", "")
        if "MVE" in ticker or "MULTIGAME" in ticker:
            return True
        return False

    def _normalize(self, m: dict) -> UnifiedMarket:
        # ── Price extraction ──────────────────────────────────────────
        # Kalshi migrated from integer-cent fields (yes_bid, no_bid, etc.)
        # to fixed-point dollar strings (yes_bid_dollars, no_bid_dollars).
        # The legacy cent fields were removed March 12 2026, but some
        # cached responses or SDK versions may still return them.
        # We try _dollars first, then fall back to cents ÷ 100.
        yes_price = self._get_price(m, "yes_bid_dollars", "yes_bid")
        no_price = self._get_price(m, "no_bid_dollars", "no_bid")
        last_price = self._get_price(m, "last_price_dollars", "last_price")
        yes_ask = self._get_price(m, "yes_ask_dollars", "yes_ask")

        logger.debug(
            "Kalshi price fields for %s: yes_bid_dollars=%r yes_bid=%r",
            m.get("ticker"),
            m.get("yes_bid_dollars"),
            m.get("yes_bid"),
        )

        # ── Outcome labels ────────────────────────────────────────────
        # Kalshi uses yes_sub_title / no_sub_title inconsistently:
        #   - Standard markets: "Yes" / "No"
        #   - Sports props: "Jarrett Allen: 15+" / "Jarrett Allen: 15+"
        #     (both sides get the SAME prop description)
        #   - Range markets: "Above 55°F" / "Below 55°F"
        # For binary contracts, normalize to "Yes"/"No" when the
        # sub_titles are identical or match the title (the prop
        # semantics are already captured in the question field).
        yes_label = m.get("yes_sub_title", "Yes") or "Yes"
        no_label = m.get("no_sub_title", "No") or "No"
        if yes_label == no_label:
            # Both sides have the same label (sports props) — meaningless
            yes_label, no_label = "Yes", "No"

        outcomes = [
            Outcome(label=yes_label, price=yes_price, index=0),
            Outcome(label=no_label, price=no_price, index=1),
        ]

        # ── Volume / size extraction ──────────────────────────────────
        # Same migration: volume_fp (string) vs volume (int).
        volume_total = self._get_count(m, "volume_fp", "volume")
        volume_24h = self._get_count(m, "volume_24h_fp", "volume_24h")
        open_interest = self._get_count(m, "open_interest_fp", "open_interest")

        # ── Spread ────────────────────────────────────────────────────
        spread = None
        if yes_ask is not None and yes_price is not None:
            spread = round(yes_ask - yes_price, 4)

        # ── Text fields ───────────────────────────────────────────────
        # rules_primary contains the resolution criteria.  Use it as
        # resolution_rules, and use subtitle (if present and different
        # from title) as a lightweight description to avoid duplication.
        title = m.get("title") or m.get("ticker", "")
        rules = m.get("rules_primary", "")
        subtitle = m.get("subtitle", "")
        # Use subtitle as description only if it adds information
        description = subtitle if subtitle and subtitle != title else ""

        return UnifiedMarket(
            exchange=Exchange.KALSHI,
            native_id=m.get("ticker", ""),
            slug=m.get("ticker", ""),
            question=title,
            description=description,
            category=m.get("category", ""),
            event_id=m.get("event_ticker"),
            event_title=m.get("event_ticker"),  # Need separate event fetch for title
            group_id=m.get("series_ticker"),
            outcome_type=OutcomeType.BINARY,
            outcomes=outcomes,
            yes_price=yes_price,
            no_price=no_price,
            last_price=last_price,
            spread=spread,
            volume_total=volume_total,
            volume_24h=volume_24h,
            liquidity=self._get_price(m, "liquidity_dollars", None),
            open_interest=open_interest,
            created_at=self._parse_ts(m.get("created_time")),
            open_at=self._parse_ts(m.get("open_time")),
            close_at=self._parse_ts(m.get("close_time")),
            status=self.STATUS_MAP.get(m.get("status", ""), MarketStatus.UNKNOWN),
            resolution_source=ResolutionSource.CENTRALIZED_ORACLE,
            resolution_rules=rules,
            result=m.get("result"),
            url=f"https://kalshi.com/markets/{m.get('ticker', '')}",
            fetched_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _get_price(m: dict, dollars_key: str, cents_key: Optional[str]) -> Optional[float]:
        """
        Extract a price, preferring the _dollars string field,
        falling back to the legacy integer-cents field ÷ 100.
        Returns None if the value is absent or zero (no bids posted).
        """
        # Try the _dollars field first (string like "0.5600")
        raw = m.get(dollars_key)
        if raw is not None:
            try:
                val = float(raw)
                return val if val > 0 else None
            except (ValueError, TypeError):
                pass

        # Fall back to legacy integer cents field
        if cents_key is not None:
            raw = m.get(cents_key)
            if raw is not None:
                try:
                    val = int(raw)
                    return val / 100.0 if val > 0 else None
                except (ValueError, TypeError):
                    pass

        return None

    @staticmethod
    def _get_count(m: dict, fp_key: str, int_key: Optional[str]) -> Optional[float]:
        """
        Extract a count/volume, preferring the _fp string field,
        falling back to the legacy integer field.
        Returns None only if both are absent; 0.0 is valid for volume.
        """
        raw = m.get(fp_key)
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass

        if int_key is not None:
            raw = m.get(int_key)
            if raw is not None:
                try:
                    return float(int(raw))
                except (ValueError, TypeError):
                    pass

        return None


# ─── Polymarket Adapter ──────────────────────────────────────────────────────

class PolymarketAdapter(ExchangeAdapter):
    """
    Polymarket Gamma API (public, no auth).

    Base URL: https://gamma-api.polymarket.com
    Hierarchy: Event → Market (each market has Yes/No token pair)
    Pagination: offset-based (limit + offset params)
    """

    BASE = "https://gamma-api.polymarket.com"

    async def fetch_markets(
        self,
        status: Optional[str] = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[UnifiedMarket]:
        offset = int(cursor) if cursor else 0
        page_size = min(limit, 100)

        while True:
            params = {"limit": page_size, "offset": offset}
            if status == "open":
                params["closed"] = "false"
                params["active"] = "true"

            resp = await self.client.get(f"{self.BASE}/markets", params=params)
            resp.raise_for_status()
            markets = resp.json()

            if not markets:
                break

            for m in markets:
                yield self._normalize(m)

            if len(markets) < page_size:
                break
            offset += page_size

    def _normalize(self, m: dict) -> UnifiedMarket:
        # Parse outcomes and prices from JSON-encoded strings
        outcome_labels = self._parse_json_array(m.get("outcomes", "[]"))
        outcome_prices = self._parse_json_array(m.get("outcomePrices", "[]"))

        outcomes = []
        for i, label in enumerate(outcome_labels):
            price = float(outcome_prices[i]) if i < len(outcome_prices) else None
            token_ids = self._parse_json_array(m.get("clobTokenIds", "[]"))
            token_id = token_ids[i] if i < len(token_ids) else None
            outcomes.append(Outcome(label=label, price=price, token_id=token_id, index=i))

        yes_price = float(outcome_prices[0]) if outcome_prices else None
        no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else None

        # Determine outcome type
        outcome_type = OutcomeType.BINARY if len(outcomes) <= 2 else OutcomeType.MULTI

        return UnifiedMarket(
            exchange=Exchange.POLYMARKET,
            native_id=m.get("conditionId", m.get("id", "")),
            slug=m.get("slug"),
            question=m.get("question", m.get("title", "")),
            description=m.get("description", ""),
            category=m.get("groupItemTitle", ""),
            tags=self._extract_tags(m),
            event_id=m.get("eventSlug"),
            event_title=m.get("groupItemTitle"),
            group_id=m.get("eventSlug"),
            outcome_type=outcome_type,
            outcomes=outcomes,
            yes_price=yes_price,
            no_price=no_price,
            last_price=yes_price,
            volume_total=self._safe_float(m.get("volume")),
            volume_24h=self._safe_float(m.get("volume24hr")),
            liquidity=self._safe_float(m.get("liquidity")),
            created_at=self._parse_ts(m.get("createdAt")),
            close_at=self._parse_ts(m.get("endDate")),
            status=self._map_status(m),
            resolution_source=ResolutionSource.UMA_ORACLE,
            resolution_rules=m.get("resolutionSource", ""),
            result=m.get("resolvedBy"),
            url=f"https://polymarket.com/event/{m.get('slug', '')}",
            fetched_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _parse_json_array(val) -> list:
        import json
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_tags(m: dict) -> list[str]:
        tags = []
        if m.get("tags"):
            if isinstance(m["tags"], list):
                tags = m["tags"]
            elif isinstance(m["tags"], str):
                tags = [t.strip() for t in m["tags"].split(",")]
        return tags

    @staticmethod
    def _map_status(m: dict) -> MarketStatus:
        if m.get("resolved"):
            return MarketStatus.SETTLED
        if m.get("closed"):
            return MarketStatus.CLOSED
        if m.get("active"):
            return MarketStatus.OPEN
        return MarketStatus.UNKNOWN


# ─── Manifold Adapter ────────────────────────────────────────────────────────

class ManifoldAdapter(ExchangeAdapter):
    """
    Manifold Markets API (public, no auth for reads).

    Base URL: https://api.manifold.markets/v0
    Play-money market. Useful for semantic coverage and testing.
    Pagination: cursor-based (before parameter)
    """

    BASE = "https://api.manifold.markets/v0"

    async def fetch_markets(
        self,
        status: Optional[str] = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[UnifiedMarket]:
        params = {"limit": min(limit, 1000)}
        if cursor:
            params["before"] = cursor

        resp = await self.client.get(f"{self.BASE}/markets", params=params)
        resp.raise_for_status()
        markets = resp.json()

        for m in markets:
            if status == "open" and m.get("isResolved", False):
                continue
            if m.get("outcomeType") not in ("BINARY", "MULTIPLE_CHOICE"):
                continue
            yield self._normalize(m)

    def _normalize(self, m: dict) -> UnifiedMarket:
        prob = m.get("probability")
        outcomes = []
        if m.get("outcomeType") == "BINARY":
            outcomes = [
                Outcome(label="Yes", price=prob, index=0),
                Outcome(label="No", price=(1.0 - prob) if prob else None, index=1),
            ]
            outcome_type = OutcomeType.BINARY
        else:
            outcome_type = OutcomeType.MULTI
            for ans in m.get("answers", []):
                outcomes.append(Outcome(
                    label=ans.get("text", ""),
                    price=ans.get("probability"),
                    index=ans.get("index", 0),
                ))

        return UnifiedMarket(
            exchange=Exchange.MANIFOLD,
            native_id=m.get("id", ""),
            slug=m.get("slug"),
            question=m.get("question", ""),
            description=m.get("textDescription", ""),
            category=m.get("groupSlugs", [""])[0] if m.get("groupSlugs") else "",
            tags=m.get("groupSlugs", []),
            outcome_type=outcome_type,
            outcomes=outcomes,
            yes_price=prob,
            no_price=(1.0 - prob) if prob else None,
            last_price=prob,
            volume_total=m.get("volume"),
            volume_24h=m.get("volume24Hours"),
            liquidity=m.get("totalLiquidity"),
            created_at=self._parse_ts(m.get("createdTime")),
            close_at=self._parse_ts(m.get("closeTime")),
            status=MarketStatus.SETTLED if m.get("isResolved") else MarketStatus.OPEN,
            resolution_source=ResolutionSource.COMMUNITY,
            result=m.get("resolution"),
            url=m.get("url", f"https://manifold.markets/{m.get('creatorUsername')}/{m.get('slug')}"),
            fetched_at=datetime.now(timezone.utc),
        )


# ─── Metaculus Adapter ───────────────────────────────────────────────────────

class MetaculusAdapter(ExchangeAdapter):
    """
    Metaculus API (public for reads).

    Base URL: https://www.metaculus.com/api2
    Not a trading market—forecasting platform with crowd aggregation.
    Useful for semantic matching against real-money markets.
    """

    BASE = "https://www.metaculus.com/api2"

    async def fetch_markets(
        self,
        status: Optional[str] = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> AsyncIterator[UnifiedMarket]:
        params = {"limit": min(limit, 100), "type": "forecast"}
        if status == "open":
            params["status"] = "open"
        if cursor:
            params["offset"] = int(cursor)

        resp = await self.client.get(
            f"{self.BASE}/questions/",
            params=params,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        for q in data.get("results", []):
            market = self._normalize(q)
            if market:
                yield market

    def _normalize(self, q: dict) -> Optional[UnifiedMarket]:
        community_pred = q.get("community_prediction", {})
        prob = None
        if isinstance(community_pred, dict):
            prob = community_pred.get("full", {}).get("q2")  # Median
        elif isinstance(community_pred, (int, float)):
            prob = float(community_pred)

        # Determine outcome type
        possibilities = q.get("possibilities", {})
        qtype = possibilities.get("type", "binary")
        if qtype == "binary":
            outcome_type = OutcomeType.BINARY
            outcomes = [
                Outcome(label="Yes", price=prob, index=0),
                Outcome(label="No", price=(1.0 - prob) if prob else None, index=1),
            ]
        elif qtype in ("continuous", "date"):
            outcome_type = OutcomeType.SCALAR
            outcomes = []
        else:
            outcome_type = OutcomeType.BINARY
            outcomes = []

        return UnifiedMarket(
            exchange=Exchange.METACULUS,
            native_id=str(q.get("id", "")),
            slug=q.get("url_title"),
            question=q.get("title", ""),
            description=q.get("description", ""),
            category=q.get("group", ""),
            tags=[t.get("name", "") for t in q.get("tags", []) if isinstance(t, dict)],
            outcome_type=outcome_type,
            outcomes=outcomes,
            yes_price=prob,
            no_price=(1.0 - prob) if prob else None,
            last_price=prob,
            created_at=self._parse_ts(q.get("created_time")),
            close_at=self._parse_ts(q.get("close_time")),
            resolved_at=self._parse_ts(q.get("resolve_time")),
            status=self._map_status(q),
            resolution_source=ResolutionSource.CROWD_FORECAST,
            resolution_rules=q.get("resolution_criteria", ""),
            url=f"https://www.metaculus.com/questions/{q.get('id')}",
            fetched_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _map_status(q: dict) -> MarketStatus:
        s = q.get("active_state", q.get("status", ""))
        if s in ("RESOLVED", "CLOSED"):
            return MarketStatus.SETTLED
        if s in ("OPEN", "UPCOMING"):
            return MarketStatus.OPEN
        return MarketStatus.UNKNOWN