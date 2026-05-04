#!/usr/bin/env python3
"""
Backtest our ArbitrageCalculator against PredictionMarketBench episode data.

Within each Kalshi episode, multi-outcome events contain COMPLEMENT ticker pairs
(e.g., BUF-win vs JAC-win). When their YES prices sum to < $1.00, buying both
is a risk-free hedge. This agent detects those gaps and places both legs.

Usage:
    python run_bench_arb.py
    python run_bench_arb.py --episodes KXNFLGAME-26JAN11BUFJAC
    python run_bench_arb.py --fee-rate 0.01 --min-profit-cents 2 --cadence 30
"""

from __future__ import annotations

import os
import sys
import argparse
from itertools import combinations
from pathlib import Path

# Resolve repo roots from this file's location
_REPO_ROOT = Path(__file__).resolve().parent
_BENCH_SRC = Path(os.environ.get("BENCH_SRC", str(_REPO_ROOT.parent / "PredictionMarketBench" / "src")))
_PIPELINE = _REPO_ROOT / "UnifiedMarketPipeline"

sys.path.insert(0, str(_BENCH_SRC))
sys.path.insert(0, str(_PIPELINE))

from oddpool_bench import (
    BenchmarkHarness,
    SimulatorConfig,
    Agent,
    AgentContext,
    Order,
    Side,
    Action,
    OrderType,
)
from arbitrage_calculator import ArbitrageCalculator
from models import UnifiedMarket, Exchange


def _to_unified(ticker: str, yes_ask_cents: int, no_ask_cents: int) -> UnifiedMarket:
    """Create a minimal UnifiedMarket from Kalshi orderbook ask prices."""
    yes_price = yes_ask_cents / 100.0
    no_price = no_ask_cents / 100.0
    return UnifiedMarket(
        exchange=Exchange.KALSHI,
        native_id=ticker,
        question=ticker,
        yes_price=yes_price,
        no_price=no_price,
    )


class ArbitrageAgent(Agent):
    """
    Detects COMPLEMENT hedge opportunities between tickers in the same episode.

    Within a Kalshi event, multi-ticker outcomes (e.g., team A wins / team B wins)
    are COMPLEMENT markets. The ArbitrageCalculator flags HEDGE_ARB when
    P(A YES) + P(B YES) < 1.0 — guaranteed profit if we buy YES on both.
    """

    def __init__(
        self,
        fee_rate: float = 0.0,
        min_profit_cents: float = 2.0,
        max_contracts: int = 5,
    ):
        self.calculator = ArbitrageCalculator(default_fee_rate=fee_rate)
        self.min_profit_cents = min_profit_cents
        self.max_contracts = max_contracts

    def on_episode_start(self, metadata: dict) -> None:
        pass

    def act(self, ctx: AgentContext) -> None:
        markets = ctx.get_markets()
        positions = ctx.get_positions()
        cash = ctx.get_cash()

        # Only consider markets with full quotes on both sides
        active = [
            m for m in markets
            if (
                m.yes_best_bid is not None
                and m.yes_best_ask is not None
                and m.no_best_bid is not None
                and m.no_best_ask is not None
            )
        ]
        if len(active) < 2:
            return

        for m_a, m_b in combinations(active, 2):
            # Skip if we already hold a position on either side
            if self._has_position(positions, m_a.ticker) or self._has_position(positions, m_b.ticker):
                continue

            # Need enough cash for at least 1 contract each
            if cash["cash_cents"] < (m_a.yes_best_ask + m_b.yes_best_ask):
                continue

            u_a = _to_unified(m_a.ticker, m_a.yes_best_ask, m_a.no_best_ask)
            u_b = _to_unified(m_b.ticker, m_b.yes_best_ask, m_b.no_best_ask)

            opportunities = self.calculator.evaluate_pair(u_a, u_b, "COMPLEMENT")
            if not opportunities:
                continue

            for opp in opportunities:
                if opp.type != "HEDGE_ARB":
                    continue
                profit_cents = opp.net_profit * 100.0
                if profit_cents < self.min_profit_cents:
                    continue

                # Determine which sides to buy
                if "YES" in opp.leg1_action:
                    side_a = Side.YES
                    price_a = m_a.yes_best_ask
                    side_b = Side.YES
                    price_b = m_b.yes_best_ask
                else:
                    side_a = Side.NO
                    price_a = m_a.no_best_ask
                    side_b = Side.NO
                    price_b = m_b.no_best_ask

                # Size position: how many contracts fit in cash
                max_by_cash = cash["cash_cents"] // (price_a + price_b)
                count = min(int(max_by_cash), self.max_contracts)
                if count < 1:
                    continue

                ctx.place_order(Order(
                    ticker=m_a.ticker,
                    side=side_a,
                    action=Action.BUY,
                    order_type=OrderType.LIMIT,
                    count=count,
                    limit_price_cents=price_a,
                ))
                ctx.place_order(Order(
                    ticker=m_b.ticker,
                    side=side_b,
                    action=Action.BUY,
                    order_type=OrderType.LIMIT,
                    count=count,
                    limit_price_cents=price_b,
                ))
                break  # one trade pair per step

    @staticmethod
    def _has_position(positions: dict, ticker: str) -> bool:
        pos = positions.get(ticker, {})
        return pos.get("yes_contracts", 0) != 0 or pos.get("no_contracts", 0) != 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArbitrageAgent on PredictionMarketBench episodes.")
    parser.add_argument("--episodes", nargs="*", help="Episode IDs to run (default: all)")
    parser.add_argument("--fee-rate", type=float, default=0.0,
                        help="Fee rate as decimal, e.g. 0.01 = 1%% (default: 0)")
    parser.add_argument("--min-profit-cents", type=float, default=2.0,
                        help="Minimum net profit in cents to enter a trade (default: 2)")
    parser.add_argument("--max-contracts", type=int, default=5,
                        help="Maximum contracts per leg per trade (default: 5)")
    parser.add_argument("--cadence", type=float, default=30.0,
                        help="Agent call interval in seconds (default: 30)")
    parser.add_argument("--output-dir", type=str, default="bench_results",
                        help="Directory to write results (default: bench_results)")
    args = parser.parse_args()

    episodes_dir = _BENCH_SRC.parent / "episodes"
    config = SimulatorConfig(
        agent_call_cadence_seconds=args.cadence,
        equity_sample_interval_seconds=60.0,
        verbose=True,
    )

    harness = BenchmarkHarness(episodes_dir, config)
    episodes = args.episodes or harness.list_episodes()
    print(f"\nEpisodes to run: {episodes}")

    agent = ArbitrageAgent(
        fee_rate=args.fee_rate,
        min_profit_cents=args.min_profit_cents,
        max_contracts=args.max_contracts,
    )
    result = harness.run(agent, episode_ids=episodes)
    result.print_summary()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    result.save(output_dir / "summary.json")
    result.save_trades(output_dir / "trades.json")
    result.save_equity_csv(output_dir / "equity_curve.csv")
    try:
        result.save_equity_curve(output_dir / "equity_curve.png")
        print(f"Equity curve saved to {output_dir}/equity_curve.png")
    except ImportError:
        print("(Skipping plot — run: pip install matplotlib)")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
