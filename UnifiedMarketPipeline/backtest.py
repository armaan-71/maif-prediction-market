"""
Backtest the cross-exchange arbitrage algorithm against historical data.

Reads verified pairs from verified_pairs.json, pulls historical price series
for each market via HistoricalDataManager (Polymarket native API + Oddpool for
Kalshi), then replays the algorithm timestep-by-timestep:

  for each timestamp:
      for each pair:
          if no open trade and ArbitrageCalculator finds an opportunity:
              enter
      for each open trade:
          if exit condition met:
              close, record P&L

Usage:
    python backtest.py --pairs verified_pairs.json \
        --start 2025-01-01 --end 2025-12-31 \
        [--fidelity 60] [--fee-rate 0.01] [--min-roi 0.005] \
        [--position-size 100] [--no-fetch]
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import UnifiedMarket, Exchange, MarketStatus
from arbitrage_calculator import ArbitrageCalculator, ArbitrageOpportunity
from historical_data import (
    HistoricalDataManager,
    SUPPORTED_EXCHANGES,
    load_pairs,
    markets_from_pairs,
)

from pipeline import setup_logging

logger = logging.getLogger("backtest")


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    pair_id: str
    relation: str
    opp_type: str

    entry_time: datetime
    leg1_uid: str
    leg1_action: str
    leg1_entry_price: float
    leg2_uid: str
    leg2_action: str
    leg2_entry_price: float

    entry_cost: float               # gross cost per unit (before fees)
    entry_cost_after_fees: float    # net cost per unit (after fees)
    position_size: float

    initial_gap: float = 0.0  # for DIRECTIONAL_GAP exit logic

    exit_time: Optional[datetime] = None
    realized_pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    status: str = "open"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat() if self.exit_time else None
        return d


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)

    @property
    def closed(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "closed"]

    @property
    def open(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "open"]

    @property
    def total_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.closed if t.realized_pnl is not None)

    @property
    def total_deployed(self) -> float:
        """Capital deployed across all trades (open + closed)."""
        return sum(t.entry_cost_after_fees * t.position_size for t in self.trades)

    @property
    def total_closed_invested(self) -> float:
        """Capital deployed in closed trades only."""
        return sum(t.entry_cost_after_fees * t.position_size for t in self.closed)

    @property
    def realized_roi(self) -> float:
        """ROI on closed trades only — the most meaningful metric while positions are open."""
        return self.total_pnl / self.total_closed_invested if self.total_closed_invested else 0.0

    @property
    def deployed_roi(self) -> float:
        """ROI across all deployed capital including open (unrealized) positions."""
        return self.total_pnl / self.total_deployed if self.total_deployed else 0.0

    @property
    def win_rate(self) -> float:
        closed = self.closed
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if (t.realized_pnl or 0) > 0)
        return wins / len(closed)

    @property
    def avg_duration_hours(self) -> float:
        closed = self.closed
        if not closed:
            return 0.0
        durs = [
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in closed if t.exit_time
        ]
        return sum(durs) / len(durs) if durs else 0.0

    def by_opp_type(self) -> dict[str, dict]:
        out: dict[str, dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
        for t in self.closed:
            out[t.opp_type]["count"] += 1
            out[t.opp_type]["pnl"] += t.realized_pnl or 0
        return dict(out)

    def summary(self) -> str:
        lines = [
            "─" * 60,
            "BACKTEST SUMMARY",
            "─" * 60,
            f"Total trades:       {len(self.trades)} "
            f"(closed: {len(self.closed)}, open: {len(self.open)})",
            f"Total P&L:          ${self.total_pnl:.2f}",
            f"Total deployed:     ${self.total_deployed:.2f}",
            f"Realized ROI:       {self.realized_roi*100:.2f}%  (closed trades only)",
            f"Deployed ROI:       {self.deployed_roi*100:.2f}%  (all capital incl. open)",
            f"Win rate:           {self.win_rate*100:.2f}%",
            f"Avg duration:       {self.avg_duration_hours:.1f} h",
        ]
        breakdown = self.by_opp_type()
        if breakdown:
            lines.append("By opportunity type:")
            for k, v in breakdown.items():
                lines.append(f"  {k:20s}  count={v['count']:3d}  pnl=${v['pnl']:.2f}")
        lines.append("─" * 60)
        return "\n".join(lines)


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(
        self,
        fee_rate: float = 0.0,
        min_roi: float = 0.0,
        min_gap: float = 0.05,
        gap_close_threshold: float = 0.5,
        position_size: float = 1.0,
    ):
        self.calculator = ArbitrageCalculator(default_fee_rate=fee_rate)
        self.min_roi = min_roi
        self.min_gap = min_gap
        self.gap_close_threshold = gap_close_threshold
        self.position_size = position_size

    # ── public ────────────────────────────────────────────────────────────

    def run(
        self,
        pairs: list[dict],
        snapshots: dict[str, list[UnifiedMarket]],
    ) -> BacktestResult:
        """
        pairs: list of verified-pair dicts (relation in {IDENTICAL, COMPLEMENT}).
        snapshots: uid → list of UnifiedMarket sorted ascending by fetched_at.
        """
        result = BacktestResult()

        # Pre-sort snapshots and build a parallel timestamp list per uid
        ts_index: dict[str, list[datetime]] = {}
        for uid, snaps in snapshots.items():
            snaps.sort(key=lambda m: m.fetched_at or datetime.min.replace(tzinfo=timezone.utc))
            ts_index[uid] = [m.fetched_at for m in snaps]

        # Filter pairs to those we can actually price
        usable_pairs = [
            p for p in pairs
            if p["market_a"]["uid"] in snapshots
            and p["market_b"]["uid"] in snapshots
            and snapshots[p["market_a"]["uid"]]
            and snapshots[p["market_b"]["uid"]]
        ]
        if not usable_pairs:
            logger.warning("No pairs have snapshot data on both legs.")
            return result
        logger.info(f"Backtesting {len(usable_pairs)} of {len(pairs)} pairs")

        # Union of all timestamps
        all_ts = sorted({
            t for uid in {p["market_a"]["uid"] for p in usable_pairs}
                  | {p["market_b"]["uid"] for p in usable_pairs}
            for t in ts_index.get(uid, [])
        })

        open_trades: dict[str, Trade] = {}  # pair_id → Trade

        for ts in all_ts:
            # 1. Try to close existing trades.
            for pair_id, trade in list(open_trades.items()):
                pair = next(p for p in usable_pairs if p["pair_id"] == pair_id)
                m_a = self._snap_at(snapshots, ts_index, pair["market_a"]["uid"], ts)
                m_b = self._snap_at(snapshots, ts_index, pair["market_b"]["uid"], ts)
                if m_a and m_b and self._try_close(trade, m_a, m_b, ts):
                    del open_trades[pair_id]

            # 2. Try to enter new trades.
            for pair in usable_pairs:
                if pair["pair_id"] in open_trades:
                    continue
                m_a = self._snap_at(snapshots, ts_index, pair["market_a"]["uid"], ts)
                m_b = self._snap_at(snapshots, ts_index, pair["market_b"]["uid"], ts)
                if not m_a or not m_b:
                    continue
                if m_a.status == MarketStatus.SETTLED or m_b.status == MarketStatus.SETTLED:
                    continue
                trade = self._try_open(pair, m_a, m_b, ts)
                if trade:
                    open_trades[pair["pair_id"]] = trade
                    result.trades.append(trade)

        # 3. Window expired: leftover trades stay status="open".
        return result

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _snap_at(
        snapshots: dict[str, list[UnifiedMarket]],
        ts_index: dict[str, list[datetime]],
        uid: str,
        ts: datetime,
    ) -> Optional[UnifiedMarket]:
        """Most recent snapshot of `uid` at or before `ts`."""
        timeline = ts_index.get(uid)
        if not timeline:
            return None
        idx = bisect.bisect_right(timeline, ts) - 1
        return snapshots[uid][idx] if idx >= 0 else None

    def _try_open(
        self,
        pair: dict,
        m_a: UnifiedMarket,
        m_b: UnifiedMarket,
        ts: datetime,
    ) -> Optional[Trade]:
        relation = pair.get("relation")
        if relation not in ("IDENTICAL", "COMPLEMENT"):
            return None
        if m_a.yes_price is None or m_b.yes_price is None:
            return None

        opportunities = self.calculator.evaluate_pair(m_a, m_b, relation)
        if not opportunities:
            return None

        # Prefer the cheapest HEDGE_ARB above min_roi.
        hedges = [o for o in opportunities
                  if o.type == "HEDGE_ARB" and o.roi >= self.min_roi]
        if hedges:
            best = min(hedges, key=lambda o: o.total_cost)
            return self._make_hedge_trade(pair, best, m_a, m_b, ts)

        # Otherwise, the largest DIRECTIONAL_GAP above min_gap (IDENTICAL only).
        gaps = [o for o in opportunities
                if o.type == "DIRECTIONAL_GAP" and o.net_profit >= self.min_gap]
        if gaps:
            best = max(gaps, key=lambda o: o.net_profit)
            return self._make_gap_trade(pair, best, m_a, m_b, ts)

        return None

    def _make_hedge_trade(
        self, pair: dict, opp: ArbitrageOpportunity,
        m_a: UnifiedMarket, m_b: UnifiedMarket, ts: datetime,
    ) -> Trade:
        return Trade(
            pair_id=pair["pair_id"],
            relation=pair["relation"],
            opp_type="HEDGE_ARB",
            entry_time=ts,
            leg1_uid=m_a.uid,
            leg1_action=opp.leg1_action,
            leg1_entry_price=opp.leg1_price,
            leg2_uid=m_b.uid,
            leg2_action=opp.leg2_action,
            leg2_entry_price=opp.leg2_price,
            entry_cost=opp.total_cost,
            entry_cost_after_fees=opp.total_cost_after_fees,
            position_size=self.position_size,
        )

    def _make_gap_trade(
        self, pair: dict, opp: ArbitrageOpportunity,
        m_a: UnifiedMarket, m_b: UnifiedMarket, ts: datetime,
    ) -> Trade:
        # Only buy the cheaper YES leg.
        if opp.leg1_price <= opp.leg2_price:
            buy_uid, buy_price, other_uid, other_price = (
                m_a.uid, opp.leg1_price, m_b.uid, opp.leg2_price
            )
        else:
            buy_uid, buy_price, other_uid, other_price = (
                m_b.uid, opp.leg2_price, m_a.uid, opp.leg1_price
            )
        return Trade(
            pair_id=pair["pair_id"],
            relation=pair["relation"],
            opp_type="DIRECTIONAL_GAP",
            entry_time=ts,
            leg1_uid=buy_uid,
            leg1_action="BUY YES",
            leg1_entry_price=buy_price,
            leg2_uid="",
            leg2_action="",
            leg2_entry_price=0.0,
            entry_cost=buy_price,
            entry_cost_after_fees=buy_price,  # no fee model for directional legs
            position_size=self.position_size,
            initial_gap=abs(other_price - buy_price),
        )

    def _try_close(
        self,
        trade: Trade,
        m_a: UnifiedMarket,
        m_b: UnifiedMarket,
        ts: datetime,
    ) -> bool:
        m_by_uid = {m_a.uid: m_a, m_b.uid: m_b}

        if trade.opp_type == "HEDGE_ARB":
            # Both legs must settle for the guarantee to materialize.
            settled = (
                m_a.status == MarketStatus.SETTLED
                and m_b.status == MarketStatus.SETTLED
            )
            if not settled:
                return False
            # Hedge is mathematically guaranteed to pay 1.0 per unit.
            pnl = (1.0 - trade.entry_cost_after_fees) * trade.position_size
            self._close(trade, ts, pnl, "resolved")
            return True

        # DIRECTIONAL_GAP — single leg
        leg_market = m_by_uid.get(trade.leg1_uid)
        if leg_market is None:
            return False

        if leg_market.status == MarketStatus.SETTLED:
            won = (leg_market.result or "").strip().lower() in ("yes", "y", "true", "1")
            payout = 1.0 if won else 0.0
            pnl = (payout - trade.leg1_entry_price) * trade.position_size
            self._close(trade, ts, pnl, "resolved")
            return True

        # Early exit if the gap has closed enough.
        other_uid = m_b.uid if trade.leg1_uid == m_a.uid else m_a.uid
        other = m_by_uid[other_uid]
        if (
            leg_market.yes_price is not None
            and other.yes_price is not None
            and trade.initial_gap > 0
        ):
            current_gap = abs(other.yes_price - leg_market.yes_price)
            if current_gap <= trade.initial_gap * (1 - self.gap_close_threshold):
                pnl = (leg_market.yes_price - trade.leg1_entry_price) * trade.position_size
                self._close(trade, ts, pnl, "converged")
                return True
        return False

    @staticmethod
    def _close(trade: Trade, ts: datetime, pnl: float, reason: str) -> None:
        trade.exit_time = ts
        trade.realized_pnl = pnl
        trade.exit_reason = reason
        trade.status = "closed"


# ─── Loader ───────────────────────────────────────────────────────────────────

def build_snapshots(
    pairs: list[dict],
    start: datetime,
    end: datetime,
    fidelity_minutes: int,
    refresh: bool,
) -> dict[str, list[UnifiedMarket]]:
    markets = markets_from_pairs(pairs)
    mgr = HistoricalDataManager()
    snapshots: dict[str, list[UnifiedMarket]] = {}
    try:
        for m in markets:
            if m.exchange not in SUPPORTED_EXCHANGES:
                logger.info(f"skip {m.uid} (exchange {m.exchange.value} unsupported)")
                continue
            snaps = mgr.materialize_snapshots(m, start, end, fidelity_minutes, refresh)
            snapshots[m.uid] = snaps
            logger.info(f"{m.uid}: {len(snaps)} snapshots")
    finally:
        mgr.close()
    return snapshots


def _parse_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Backtest the arbitrage algorithm.")
    parser.add_argument("--pairs", required=True, help="verified_pairs.json")
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2025-01-01")
    parser.add_argument("--end", required=True, help="ISO date, e.g. 2025-12-31")
    parser.add_argument("--fidelity", type=int, default=60)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--min-roi", type=float, default=0.0)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--position-size", type=float, default=1.0)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Use cache only; do not call APIs")
    parser.add_argument("--trades-out", default=None,
                        help="Write closed+open trades to this JSON file")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pairs))
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    snapshots = build_snapshots(
        pairs, start, end, args.fidelity, refresh=not args.no_fetch
    )

    engine = BacktestEngine(
        fee_rate=args.fee_rate,
        min_roi=args.min_roi,
        min_gap=args.min_gap,
        position_size=args.position_size,
    )
    result = engine.run(pairs, snapshots)

    print(result.summary())

    if args.trades_out:
        with open(args.trades_out, "w") as f:
            json.dump([t.to_dict() for t in result.trades], f, indent=2, default=str)
        print(f"Wrote {len(result.trades)} trades to {args.trades_out}")


if __name__ == "__main__":
    main()
