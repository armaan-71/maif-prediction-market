"""
Arbitrage Calculator Engine

A standalone module to evaluate pairs of prediction markets for:
1. Risk-free hedging opportunities (Arbitrage)
2. Directional price gaps

Uses theoretical prices (mid/bid) and per-exchange fee logic.
NOTE: A production version would use explicit ask prices and orderbook depth.
"""

from dataclasses import dataclass
from typing import List, Optional
from models import UnifiedMarket, Exchange


# ─── Exchange fee functions ───────────────────────────────────────────────────

def kalshi_fee_per_contract(_price: float) -> float:
    """Conservative flat fee: 7¢/contract (Kalshi's published worst-case)."""
    return 0.07


def polymarket_fee(payout: float, cost: float) -> float:
    """Polymarket charges 2% of net profit."""
    return max(0.0, payout - cost) * 0.02


def _exchange_fee(exchange: Exchange, price: float, payout: float, cost: float) -> float:
    """Dispatch to the correct per-exchange fee function."""
    if exchange == Exchange.KALSHI:
        return kalshi_fee_per_contract(price)
    if exchange == Exchange.POLYMARKET:
        return polymarket_fee(payout, cost)
    return 0.0


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class ArbitrageOpportunity:
    type: str           # "HEDGE_ARB" or "DIRECTIONAL_GAP"
    relation: str       # "IDENTICAL" or "COMPLEMENT"

    leg1_market: str
    leg1_action: str    # "BUY YES" or "BUY NO"
    leg1_price: float

    leg2_market: str
    leg2_action: str
    leg2_price: float

    total_cost: float           # gross cost before fees
    total_cost_after_fees: float = 0.0
    expected_payout: float = 1.0
    net_profit: float = 0.0     # after fees
    roi: float = 0.0            # net_profit / total_cost_after_fees

    def __repr__(self) -> str:
        return (
            f"[{self.type} | {self.relation}]\n"
            f"  Leg 1: {self.leg1_market} ({self.leg1_action} @ ${self.leg1_price:.3f})\n"
            f"  Leg 2: {self.leg2_market} ({self.leg2_action} @ ${self.leg2_price:.3f})\n"
            f"  Gross cost: ${self.total_cost:.3f}  After fees: ${self.total_cost_after_fees:.3f}\n"
            f"  Net profit: ${self.net_profit:.3f}  ROI: {self.roi*100:.2f}%"
        )


# ─── Calculator ───────────────────────────────────────────────────────────────

class ArbitrageCalculator:
    def __init__(self, default_fee_rate: float = 0.0):
        """
        default_fee_rate: flat-rate fallback when per-exchange fee functions are not defined
        for a given exchange. Per-exchange functions always take precedence.
        """
        self.default_fee_rate = default_fee_rate

    def evaluate_pair(
        self, market_a: UnifiedMarket, market_b: UnifiedMarket, relation: str
    ) -> List[ArbitrageOpportunity]:
        """
        Analyzes two markets based on their semantic relationship.
        Returns a list of identified opportunities, sorted by ROI descending.
        """
        if market_a.yes_price is None or market_b.yes_price is None:
            return []

        a_yes = market_a.yes_price
        a_no = market_a.no_price if market_a.no_price is not None else (1.0 - a_yes)
        b_yes = market_b.yes_price
        b_no = market_b.no_price if market_b.no_price is not None else (1.0 - b_yes)

        opportunities: List[ArbitrageOpportunity] = []

        if relation == "IDENTICAL":
            # Buy YES on A + Buy NO on B (locks $1 if both resolve identically)
            self._check_hedge(market_a, "YES", a_yes, market_b, "NO", b_no, relation, opportunities)
            # Buy NO on A + Buy YES on B
            self._check_hedge(market_a, "NO", a_no, market_b, "YES", b_yes, relation, opportunities)

            # Directional gap: same outcome priced differently
            gap = abs(a_yes - b_yes)
            if gap > 0.05:
                opportunities.append(ArbitrageOpportunity(
                    type="DIRECTIONAL_GAP",
                    relation=relation,
                    leg1_market=market_a.exchange.value,
                    leg1_action="BUY YES",
                    leg1_price=a_yes,
                    leg2_market=market_b.exchange.value,
                    leg2_action="BUY YES",
                    leg2_price=b_yes,
                    total_cost=0.0,
                    total_cost_after_fees=0.0,
                    net_profit=gap,
                    roi=gap / min(a_yes, b_yes),
                ))

        elif relation == "COMPLEMENT":
            # COMPLEMENT requires exhaustive outcomes (one MUST resolve YES).
            # Buy YES on A + Buy YES on B → $1 payout regardless of which resolves YES.
            self._check_hedge(market_a, "YES", a_yes, market_b, "YES", b_yes, relation, opportunities)
            # Buy NO on A + Buy NO on B
            self._check_hedge(market_a, "NO", a_no, market_b, "NO", b_no, relation, opportunities)

        return sorted(opportunities, key=lambda o: o.roi, reverse=True)

    def _check_hedge(
        self, m1: UnifiedMarket, action1: str, p1: float,
        m2: UnifiedMarket, action2: str, p2: float,
        rel: str, results: List[ArbitrageOpportunity],
    ) -> None:
        """Calculate hedge ROI with per-exchange fees. Appends to results if profitable."""
        raw_cost = p1 + p2

        # Per-exchange fees take precedence; fall back to flat rate if exchange is unknown.
        fee1 = _exchange_fee(m1.exchange, p1, 1.0, p1)
        fee2 = _exchange_fee(m2.exchange, p2, 1.0, p2)
        # If no per-exchange fee was computed (exchange not in {KALSHI, POLYMARKET}),
        # use the flat-rate fallback on the raw_cost portion.
        if fee1 == 0.0 and fee2 == 0.0 and self.default_fee_rate > 0:
            total_fees = raw_cost * self.default_fee_rate
        else:
            # Polymarket charges on profit, so compute after knowing payout.
            # Re-derive: for Polymarket payout=1.0, cost=pX.
            fee1 = _exchange_fee(m1.exchange, p1, 1.0, p1)
            fee2 = _exchange_fee(m2.exchange, p2, 1.0, p2)
            total_fees = fee1 + fee2

        total_cost_after_fees = raw_cost + total_fees

        if total_cost_after_fees < 1.0:
            profit = 1.0 - total_cost_after_fees
            roi = profit / total_cost_after_fees
            results.append(ArbitrageOpportunity(
                type="HEDGE_ARB",
                relation=rel,
                leg1_market=m1.exchange.value,
                leg1_action=f"BUY {action1}",
                leg1_price=p1,
                leg2_market=m2.exchange.value,
                leg2_action=f"BUY {action2}",
                leg2_price=p2,
                total_cost=raw_cost,
                total_cost_after_fees=total_cost_after_fees,
                expected_payout=1.0,
                net_profit=profit,
                roi=roi,
            ))
