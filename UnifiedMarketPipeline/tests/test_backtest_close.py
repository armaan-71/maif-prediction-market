"""Tests for BacktestEngine._try_close: HEDGE_ARB waits for both legs to settle."""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from backtest import BacktestEngine, Trade, BacktestResult
from models import UnifiedMarket, Exchange, MarketStatus


def _settled_market(exchange: Exchange, native_id: str) -> UnifiedMarket:
    return UnifiedMarket(
        exchange=exchange, native_id=native_id, question="q",
        yes_price=0.45, no_price=0.55,
        status=MarketStatus.SETTLED,
    )


def _open_market(exchange: Exchange, native_id: str) -> UnifiedMarket:
    return UnifiedMarket(
        exchange=exchange, native_id=native_id, question="q",
        yes_price=0.45, no_price=0.55,
        status=MarketStatus.OPEN,
    )


def _hedge_trade(uid_a: str, uid_b: str, cost_after_fees: float = 0.95) -> Trade:
    return Trade(
        pair_id="test",
        relation="IDENTICAL",
        opp_type="HEDGE_ARB",
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        leg1_uid=uid_a,
        leg1_action="BUY YES",
        leg1_entry_price=0.40,
        leg2_uid=uid_b,
        leg2_action="BUY NO",
        leg2_entry_price=0.55,
        entry_cost=0.95,
        entry_cost_after_fees=cost_after_fees,
        position_size=1.0,
    )


# ── Both legs must be settled ──────────────────────────────────────────────────

def test_hedge_not_closed_when_only_a_settles():
    engine = BacktestEngine()
    m_a = _settled_market(Exchange.KALSHI, "K1")
    m_b = _open_market(Exchange.POLYMARKET, "P1")
    trade = _hedge_trade(m_a.uid, m_b.uid)

    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    closed = engine._try_close(trade, m_a, m_b, ts)

    assert not closed
    assert trade.status == "open"


def test_hedge_not_closed_when_only_b_settles():
    engine = BacktestEngine()
    m_a = _open_market(Exchange.KALSHI, "K1")
    m_b = _settled_market(Exchange.POLYMARKET, "P1")
    trade = _hedge_trade(m_a.uid, m_b.uid)

    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    closed = engine._try_close(trade, m_a, m_b, ts)

    assert not closed
    assert trade.status == "open"


def test_hedge_closed_when_both_settle():
    engine = BacktestEngine()
    m_a = _settled_market(Exchange.KALSHI, "K1")
    m_b = _settled_market(Exchange.POLYMARKET, "P1")
    cost_after_fees = 0.95
    trade = _hedge_trade(m_a.uid, m_b.uid, cost_after_fees=cost_after_fees)

    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    closed = engine._try_close(trade, m_a, m_b, ts)

    assert closed
    assert trade.status == "closed"
    # Guaranteed $1 payout → P&L = 1.0 - cost_after_fees = 0.05
    assert abs(trade.realized_pnl - (1.0 - cost_after_fees)) < 1e-9


# ── P&L uses entry_cost_after_fees, not entry_cost ────────────────────────────

def test_hedge_pnl_uses_post_fee_cost():
    """P&L should be (1.0 - entry_cost_after_fees), not (1.0 - entry_cost)."""
    engine = BacktestEngine()
    m_a = _settled_market(Exchange.KALSHI, "K1")
    m_b = _settled_market(Exchange.POLYMARKET, "P1")

    gross_cost = 0.90
    cost_after_fees = 0.97  # fees added on top
    trade = Trade(
        pair_id="test",
        relation="IDENTICAL",
        opp_type="HEDGE_ARB",
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        leg1_uid=m_a.uid,
        leg1_action="BUY YES",
        leg1_entry_price=0.40,
        leg2_uid=m_b.uid,
        leg2_action="BUY NO",
        leg2_entry_price=0.50,
        entry_cost=gross_cost,
        entry_cost_after_fees=cost_after_fees,
        position_size=1.0,
    )

    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    engine._try_close(trade, m_a, m_b, ts)

    expected_pnl = (1.0 - cost_after_fees) * 1.0
    assert abs(trade.realized_pnl - expected_pnl) < 1e-9


# ── BacktestResult.realized_roi vs deployed_roi ───────────────────────────────

def test_realized_vs_deployed_roi():
    result = BacktestResult()

    # One closed trade: cost 0.95, pnl 0.05
    t_closed = Trade(
        pair_id="c1",
        relation="IDENTICAL",
        opp_type="HEDGE_ARB",
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        leg1_uid="k:a",
        leg1_action="BUY YES",
        leg1_entry_price=0.40,
        leg2_uid="p:b",
        leg2_action="BUY NO",
        leg2_entry_price=0.55,
        entry_cost=0.95,
        entry_cost_after_fees=0.95,
        position_size=1.0,
        status="closed",
        exit_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
        realized_pnl=0.05,
    )

    # One open trade: cost 0.96, unrealized
    t_open = Trade(
        pair_id="o1",
        relation="IDENTICAL",
        opp_type="HEDGE_ARB",
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        leg1_uid="k:c",
        leg1_action="BUY YES",
        leg1_entry_price=0.41,
        leg2_uid="p:d",
        leg2_action="BUY NO",
        leg2_entry_price=0.55,
        entry_cost=0.96,
        entry_cost_after_fees=0.96,
        position_size=1.0,
        status="open",
    )

    result.trades = [t_closed, t_open]

    # realized_roi = 0.05 / 0.95
    assert abs(result.realized_roi - 0.05 / 0.95) < 1e-9
    # deployed_roi = 0.05 / (0.95 + 0.96) = 0.05 / 1.91
    assert abs(result.deployed_roi - 0.05 / 1.91) < 1e-9
    assert result.realized_roi > result.deployed_roi
