"""
Arbitrage Calculator Engine

A standalone module to evaluate pairs of prediction markets for:
1. Risk-free hedging opportunities (Arbitrage)
2. Directional price gaps

Note: This version uses theoretical prices (mid/bid) and a generic fee rate.
A future production version would use:
- Explicit Ask prices (the actual cost to buy).
- Exchange-specific fee logic (e.g., Kalshi's per-contract fee).
- Orderbook depth (to see if enough shares are available at the current price).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from models import UnifiedMarket, Exchange

@dataclass
class ArbitrageOpportunity:
    type: str  # "HEDGE_ARB" or "DIRECTIONAL_GAP"
    relation: str # "IDENTICAL" or "COMPLEMENT"
    
    # Description of the legs
    leg1_market: str
    leg1_action: str # "BUY YES" or "BUY NO"
    leg1_price: float
    
    leg2_market: str
    leg2_action: str
    leg2_price: float
    
    total_cost: float
    expected_payout: float = 1.0
    net_profit: float = 0.0
    roi: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"[{self.type} | {self.relation}]\n"
            f"  Leg 1: {self.leg1_market} ({self.leg1_action} @ ${self.leg1_price:.3f})\n"
            f"  Leg 2: {self.leg2_market} ({self.leg2_action} @ ${self.leg2_price:.3f})\n"
            f"  Total Cost: ${self.total_cost:.3f} | Net Profit: ${self.net_profit:.3f} | ROI: {self.roi*100:.2f}%"
        )

class ArbitrageCalculator:
    def __init__(self, default_fee_rate: float = 0.0):
        """
        Initialize with a generic fee rate (e.g., 0.01 for 1%).
        In a future version, fees will be calculated dynamically based on Exchange rules.
        """
        self.default_fee_rate = default_fee_rate

    def evaluate_pair(self, market_a: UnifiedMarket, market_b: UnifiedMarket, relation: str) -> List[ArbitrageOpportunity]:
        """
        Analyzes two markets based on their semantic relationship.
        Returns a list of identified opportunities.
        """
        opportunities = []
        
        # Check if prices exist
        if market_a.yes_price is None or market_b.yes_price is None:
            return []

        # Use 1.0 - yes_price as a fallback for no_price if it's missing (common in some APIs)
        a_yes = market_a.yes_price
        a_no = market_b.no_price if market_a.no_price is not None else (1.0 - a_yes)
        
        b_yes = market_b.yes_price
        b_no = market_b.no_price if market_b.no_price is not None else (1.0 - b_yes)

        # ── 1. Hedging (Sure-Win) Opportunities ──────────────────────────────
        
        if relation == "IDENTICAL":
            # Scenario A: Buy YES on A, Buy NO on B
            self._check_hedge(market_a, "YES", a_yes, market_b, "NO", b_no, relation, opportunities)
            
            # Scenario B: Buy NO on A, Buy YES on B
            self._check_hedge(market_a, "NO", a_no, market_b, "YES", b_yes, relation, opportunities)

            # ── 2. Directional Gaps ──
            # If the prices for the same outcome differ significantly, it's a gap.
            gap = abs(a_yes - b_yes)
            if gap > 0.05: # 5% threshold
                opportunities.append(ArbitrageOpportunity(
                    type="DIRECTIONAL_GAP",
                    relation=relation,
                    leg1_market=market_a.exchange.value,
                    leg1_action="BUY YES",
                    leg1_price=a_yes,
                    leg2_market=market_b.exchange.value,
                    leg2_action="BUY YES",
                    leg2_price=b_yes,
                    total_cost=0, # Not a hedge, so cost/roi don't apply same way
                    net_profit=gap,
                    roi=gap / min(a_yes, b_yes)
                ))

        elif relation == "COMPLEMENT":
            # In COMPLEMENT, A:YES is the same as B:NO.
            # So a hedge is: Buy YES on A + Buy YES on B (or NO+NO)
            
            # Scenario A: Buy YES on A, Buy YES on B
            self._check_hedge(market_a, "YES", a_yes, market_b, "YES", b_yes, relation, opportunities)
            
            # Scenario B: Buy NO on A, Buy NO on B
            self._check_hedge(market_a, "NO", a_no, market_b, "NO", b_no, relation, opportunities)

        return opportunities

    def _check_hedge(self, m1, action1, p1, m2, action2, p2, rel, results):
        """Internal helper to calculate hedge ROI including fees."""
        
        # FUTURE VERSION: Replace p1, p2 with ASK prices.
        # FUTURE VERSION: Apply exchange-specific fee logic here.
        
        raw_cost = p1 + p2
        fee_cost = raw_cost * self.default_fee_rate
        total_cost = raw_cost + fee_cost
        
        if total_cost < 1.0:
            profit = 1.0 - total_cost
            roi = profit / total_cost
            
            results.append(ArbitrageOpportunity(
                type="HEDGE_ARB",
                relation=rel,
                leg1_market=m1.exchange.value,
                leg1_action=f"BUY {action1}",
                leg1_price=p1,
                leg2_market=m2.exchange.value,
                leg2_action=f"BUY {action2}",
                leg2_price=p2,
                total_cost=total_cost,
                net_profit=profit,
                roi=roi
            ))
