"""
Strategy Report — ranked current arbitrage opportunities.

Reads verified_pairs.json (or --pairs), loads current prices from the pairs file
(last known snapshot), and runs ArbitrageCalculator with per-exchange fees.

Usage:
    python strategy.py --pairs verified_pairs.json [--min-roi 0.0] [--top 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import UnifiedMarket, Exchange, Outcome
from arbitrage_calculator import ArbitrageCalculator, ArbitrageOpportunity
from pipeline import setup_logging


def _market_from_dict(m: dict) -> UnifiedMarket:
    exchange = Exchange(m["exchange"])
    native_id = m.get("native_id") or (
        m["uid"].split(":", 1)[1] if ":" in m["uid"] else m["uid"]
    )
    raw_outcomes = m.get("outcomes", [])
    outcomes = []
    for o in raw_outcomes:
        try:
            outcomes.append(Outcome(**o))
        except Exception:
            pass
    return UnifiedMarket(
        exchange=exchange,
        native_id=native_id,
        question=m.get("question", ""),
        yes_price=m.get("yes_price"),
        no_price=m.get("no_price"),
        url=m.get("url"),
        outcomes=outcomes,
    )


def _fmt_pct(v: float) -> str:
    return f"{v*100:+.2f}%"


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Print ranked arbitrage opportunities.")
    parser.add_argument("--pairs", default="verified_pairs.json",
                        help="Path to verified pairs JSON file (default: verified_pairs.json)")
    parser.add_argument("--min-roi", type=float, default=0.0,
                        help="Minimum net ROI to show (default: 0.0 = all profitable)")
    parser.add_argument("--top", type=int, default=20,
                        help="Maximum rows to print (default: 20)")
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"File not found: {pairs_path}")
        sys.exit(1)

    with pairs_path.open() as f:
        pairs = json.load(f)

    calc = ArbitrageCalculator()

    rows: list[dict] = []
    for pair in pairs:
        relation = pair.get("relation", "UNRELATED")
        m_a = _market_from_dict(pair["market_a"])
        m_b = _market_from_dict(pair["market_b"])

        if m_a.yes_price is None or m_b.yes_price is None:
            rows.append({
                "pair_id": pair["pair_id"],
                "relation": relation,
                "q_a": m_a.question[:45],
                "q_b": m_b.question[:45],
                "opp_type": "NO_PRICE",
                "gross_roi": None,
                "net_roi": None,
                "entry_cost": None,
                "entry_cost_after_fees": None,
            })
            continue

        opps = calc.evaluate_pair(m_a, m_b, relation)
        hedge = next((o for o in opps if o.type == "HEDGE_ARB"), None)
        gap = next((o for o in opps if o.type == "DIRECTIONAL_GAP"), None)
        best = hedge or gap

        if best is None:
            opp_type = "NONE"
            gross_roi = net_roi = None
            entry_cost = entry_after = None
        elif best.type == "HEDGE_ARB":
            opp_type = "HEDGE_ARB"
            gross_roi = (1.0 - best.total_cost) / best.total_cost if best.total_cost else 0.0
            net_roi = best.roi
            entry_cost = best.total_cost
            entry_after = best.total_cost_after_fees
        else:
            opp_type = "DIRECTIONAL_GAP"
            gross_roi = net_roi = best.roi
            entry_cost = entry_after = None

        rows.append({
            "pair_id": pair["pair_id"][-20:],
            "relation": relation,
            "q_a": m_a.question[:40],
            "q_b": m_b.question[:40],
            "opp_type": opp_type,
            "gross_roi": gross_roi,
            "net_roi": net_roi,
            "entry_cost": entry_cost,
            "entry_cost_after_fees": entry_after,
        })

    # Filter and sort
    profitable = [r for r in rows if r["net_roi"] is not None and r["net_roi"] >= args.min_roi]
    profitable.sort(key=lambda r: r["net_roi"], reverse=True)
    remaining = [r for r in rows if r not in profitable]

    print()
    print("=" * 120)
    print(f"  STRATEGY REPORT — {pairs_path}  ({len(pairs)} pairs)")
    print("=" * 120)
    print(f"  {'PAIR_ID':<22} {'REL':<11} {'TYPE':<14} {'GROSS_ROI':>10} {'NET_ROI':>10} {'COST':>7} {'COST+FEE':>9}  MARKET_A / MARKET_B")
    print("  " + "─" * 118)

    shown = 0
    for r in profitable:
        if shown >= args.top:
            break
        gross = _fmt_pct(r["gross_roi"]) if r["gross_roi"] is not None else "    N/A"
        net   = _fmt_pct(r["net_roi"])   if r["net_roi"]   is not None else "    N/A"
        cost  = f"${r['entry_cost']:.4f}"           if r["entry_cost"]             is not None else "    N/A"
        costf = f"${r['entry_cost_after_fees']:.4f}" if r["entry_cost_after_fees"] is not None else "    N/A"
        print(f"  {r['pair_id']:<22} {r['relation']:<11} {r['opp_type']:<14} {gross:>10} {net:>10} {cost:>7} {costf:>9}  {r['q_a']} / {r['q_b']}")
        shown += 1

    if not profitable:
        print("  (No opportunities meet the min-roi threshold after fees)")

    no_price = [r for r in remaining if r["opp_type"] == "NO_PRICE"]
    if no_price:
        print(f"\n  {len(no_price)} pair(s) skipped: missing yes_price on at least one leg.")

    none_opps = [r for r in remaining if r["opp_type"] == "NONE"]
    if none_opps:
        print(f"  {len(none_opps)} pair(s) have no current opportunity (spread too tight or >$1 combined cost).")

    print("=" * 120)
    print()


if __name__ == "__main__":
    main()
