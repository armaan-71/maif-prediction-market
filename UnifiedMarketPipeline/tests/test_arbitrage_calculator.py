"""Tests for ArbitrageCalculator: hedge ROI, per-exchange fees, COMPLEMENT exhaustiveness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from models import UnifiedMarket, Exchange
from arbitrage_calculator import (
    ArbitrageCalculator,
    kalshi_fee_per_contract,
    polymarket_fee,
)


def _market(exchange: Exchange, yes: float, no: float | None = None) -> UnifiedMarket:
    return UnifiedMarket(
        exchange=exchange,
        native_id="test",
        question="test",
        yes_price=yes,
        no_price=no if no is not None else (1.0 - yes),
    )


# ── kalshi_fee_per_contract ────────────────────────────────────────────────────

def test_kalshi_fee_is_flat():
    assert kalshi_fee_per_contract(0.10) == 0.07
    assert kalshi_fee_per_contract(0.90) == 0.07
    assert kalshi_fee_per_contract(0.50) == 0.07


# ── polymarket_fee ─────────────────────────────────────────────────────────────

def test_polymarket_fee_on_profit():
    # payout=1.0, cost=0.60 → profit=0.40 → fee = 0.40 * 0.02 = 0.008
    assert abs(polymarket_fee(1.0, 0.60) - 0.008) < 1e-9


def test_polymarket_fee_no_negative():
    # payout < cost → no fee
    assert polymarket_fee(0.50, 0.80) == 0.0


# ── IDENTICAL hedge: zero-fee baseline ────────────────────────────────────────

def test_identical_hedge_zero_fee():
    """BUY YES on A + BUY NO on B where a_yes=0.40, b_no=0.55 → total cost 0.95 → ROI ≈ 5.26%."""
    calc = ArbitrageCalculator(default_fee_rate=0.0)
    m_a = UnifiedMarket(
        exchange=Exchange.MANIFOLD, native_id="a", question="a", yes_price=0.40, no_price=0.60
    )
    m_b = UnifiedMarket(
        exchange=Exchange.MANIFOLD, native_id="b", question="b", yes_price=0.45, no_price=0.55
    )
    opps = calc.evaluate_pair(m_a, m_b, "IDENTICAL")
    hedges = [o for o in opps if o.type == "HEDGE_ARB"]
    assert hedges, "Expected a HEDGE_ARB opportunity"
    best = hedges[0]
    # YES-A @ 0.40 + NO-B @ 0.55 = 0.95 → profit 0.05 → ROI ≈ 5.26%
    assert abs(best.total_cost - 0.95) < 1e-9
    assert abs(best.roi - 0.05 / 0.95) < 1e-6


# ── Kalshi 7¢ fee rejects the hedge ───────────────────────────────────────────

def test_kalshi_hedge_rejected_by_fee():
    """YES @ 0.40 + NO @ 0.55 on two Kalshi markets → fees total 14¢ → total_cost_after_fees > 1.0."""
    calc = ArbitrageCalculator()
    m_a = _market(Exchange.KALSHI, yes=0.40, no=0.60)
    m_b = _market(Exchange.KALSHI, yes=0.45, no=0.55)
    opps = calc.evaluate_pair(m_a, m_b, "IDENTICAL")
    hedges = [o for o in opps if o.type == "HEDGE_ARB"]
    # 0.40 + 0.55 + 0.07 + 0.07 = 1.09 > 1.0 → should be rejected
    assert not hedges, "Kalshi double-fee should reject this hedge"


# ── COMPLEMENT: only YES+YES is the hedge ─────────────────────────────────────

def test_complement_hedge_direction():
    """COMPLEMENT hedge buys YES on both legs, not YES+NO."""
    calc = ArbitrageCalculator(default_fee_rate=0.0)
    m_a = UnifiedMarket(
        exchange=Exchange.MANIFOLD, native_id="a", question="a", yes_price=0.40, no_price=0.60
    )
    m_b = UnifiedMarket(
        exchange=Exchange.MANIFOLD, native_id="b", question="b", yes_price=0.55, no_price=0.45
    )
    opps = calc.evaluate_pair(m_a, m_b, "COMPLEMENT")
    hedges = [o for o in opps if o.type == "HEDGE_ARB"]
    assert hedges
    best = hedges[0]
    assert "YES" in best.leg1_action
    assert "YES" in best.leg2_action


# ── No opportunity when spread is too wide ────────────────────────────────────

def test_kalshi_complement_hedge_blocked_by_fees():
    """Both Kalshi legs with Kalshi 7¢ fees: YES+YES=0.97+0.14=1.11 and NO+NO=1.03+0.14=1.17 → both > 1.0."""
    calc = ArbitrageCalculator()
    m_a = _market(Exchange.KALSHI, yes=0.48, no=0.52)
    m_b = _market(Exchange.KALSHI, yes=0.49, no=0.51)
    opps = calc.evaluate_pair(m_a, m_b, "COMPLEMENT")
    assert not any(o.type == "HEDGE_ARB" for o in opps)


# ── DIRECTIONAL_GAP ───────────────────────────────────────────────────────────

def test_directional_gap_detected():
    """Price difference > 5% on IDENTICAL markets should yield a DIRECTIONAL_GAP."""
    calc = ArbitrageCalculator(default_fee_rate=0.0)
    m_a = _market(Exchange.MANIFOLD, yes=0.40)
    m_b = _market(Exchange.MANIFOLD, yes=0.50)
    opps = calc.evaluate_pair(m_a, m_b, "IDENTICAL")
    gaps = [o for o in opps if o.type == "DIRECTIONAL_GAP"]
    assert gaps
    assert abs(gaps[0].net_profit - 0.10) < 1e-9


def test_no_gap_below_threshold():
    """Price difference of ≤ 5% should not produce a DIRECTIONAL_GAP."""
    calc = ArbitrageCalculator(default_fee_rate=0.0)
    m_a = _market(Exchange.MANIFOLD, yes=0.40)
    m_b = _market(Exchange.MANIFOLD, yes=0.44)
    opps = calc.evaluate_pair(m_a, m_b, "IDENTICAL")
    assert not any(o.type == "DIRECTIONAL_GAP" for o in opps)
