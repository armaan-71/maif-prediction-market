"""
Compute arbitrage edge and strategy description from a classified market pair.
"""
from __future__ import annotations

from .models import ArbitrageOpportunity, ClassifiedPair, RelationType

MIN_CONFIDENCE = 0.70
MIN_ARB_EDGE = 0.02


def score_pair(
    pair: ClassifiedPair,
    min_confidence: float = MIN_CONFIDENCE,
    min_arb_edge: float = MIN_ARB_EDGE,
) -> ArbitrageOpportunity | None:
    if pair.confidence < min_confidence:
        return None
    if pair.relation in (RelationType.UNRELATED, RelationType.AMBIGUOUS):
        return None

    a, b = pair.market_a, pair.market_b
    ya = a.yes_price
    yb = b.yes_price

    # Can't compute edge without prices
    if ya is None or yb is None:
        return None

    arb_edge: float = 0.0
    strategy: str = ""

    if pair.relation == RelationType.IDENTICAL:
        arb_edge = abs(ya - yb)
        if ya < yb:
            strategy = (
                f"Buy YES on {a.exchange} @ {ya:.3f}, "
                f"sell YES on {b.exchange} @ {yb:.3f}"
            )
        else:
            strategy = (
                f"Buy YES on {b.exchange} @ {yb:.3f}, "
                f"sell YES on {a.exchange} @ {ya:.3f}"
            )

    elif pair.relation == RelationType.COMPLEMENT:
        # YES_A and YES_B are mutually exclusive and exhaustive: sum should be 1.0
        total = ya + yb
        if total < 1.0:
            arb_edge = 1.0 - total
            strategy = (
                f"Buy YES_A on {a.exchange} @ {ya:.3f} + "
                f"Buy YES_B on {b.exchange} @ {yb:.3f} "
                f"(guaranteed $1 payout for ${total:.3f} cost)"
            )
        elif total > 1.0:
            arb_edge = total - 1.0
            strategy = (
                f"Buy NO_A on {a.exchange} @ {1.0 - ya:.3f} + "
                f"Buy NO_B on {b.exchange} @ {1.0 - yb:.3f} "
                f"(guaranteed $1 payout for ${2.0 - total:.3f} cost)"
            )

    elif pair.relation == RelationType.SUBSET:
        # P(A) <= P(B); if ya > yb, arb exists
        if ya > yb:
            arb_edge = ya - yb
            strategy = (
                f"Sell YES_A on {a.exchange} @ {ya:.3f}, "
                f"buy YES_B on {b.exchange} @ {yb:.3f} "
                f"(A is subset of B so P(A) must be <= P(B))"
            )

    elif pair.relation == RelationType.SUPERSET:
        # P(A) >= P(B); if ya < yb, arb exists
        if ya < yb:
            arb_edge = yb - ya
            strategy = (
                f"Buy YES_A on {a.exchange} @ {ya:.3f}, "
                f"sell YES_B on {b.exchange} @ {yb:.3f} "
                f"(A is superset of B so P(A) must be >= P(B))"
            )

    elif pair.relation == RelationType.MUTUALLY_EXCLUSIVE:
        # P(A) + P(B) <= 1; if sum > 1, arb exists
        total = ya + yb
        if total > 1.0:
            arb_edge = total - 1.0
            strategy = (
                f"Sell YES_A on {a.exchange} @ {ya:.3f} + "
                f"sell YES_B on {b.exchange} @ {yb:.3f} "
                f"(mutually exclusive; sum {total:.3f} > 1)"
            )

    if arb_edge < min_arb_edge or not strategy:
        return None

    return ArbitrageOpportunity(
        pair=pair,
        arb_edge=round(arb_edge, 4),
        strategy=strategy,
    )
