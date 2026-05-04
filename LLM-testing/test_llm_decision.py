"""
Evaluation harness: runs both the rich 7-label classifier (llm_decision.py)
and the pipeline's 3-label classifier (UnifiedMarketPipeline/llm_classifier.py)
on the same test fixtures, then reports agreement.

The 7-label expected outputs in test_data.py are mapped to the 3-label space
(SUBSET / SUPERSET / MUTUALLY_EXCLUSIVE / AMBIGUOUS → expect UNRELATED) for
the pipeline comparison column.

Usage:
    python test_llm_decision.py
    python test_llm_decision.py --consensus   # 3-vote majority per comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve paths
_THIS = Path(__file__).resolve().parent
_PIPELINE = _THIS.parent / "UnifiedMarketPipeline"
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_PIPELINE))

import llm_decision as llmd
from test_data import get_test_data

data = get_test_data()

# ── Test pairs: (idx1, idx2, expected_7label) ─────────────────────────────────
PAIRS: list[tuple[int, int, str]] = [
    # IDENTICAL
    (0, 1, "IDENTICAL"),
    (6, 7, "IDENTICAL"),
    # COMPLEMENT
    (0, 2, "COMPLEMENT"),
    (9, 10, "COMPLEMENT"),
    # SUBSET / SUPERSET (threshold logic)
    (4, 0, "SUBSET"),        # 120k ⊆ 110k
    (0, 3, "SUBSET"),        # 110k ⊆ 100k
    (3, 0, "SUPERSET"),
    # TIME WINDOW SUBSET
    (5, 0, "SUBSET"),
    # OPERATOR EDGE
    (8, 0, "SUBSET"),        # >110k ⊆ >=110k
    # CPI MACRO
    (11, 12, "SUBSET"),      # >=4 ⊆ >=3
    (12, 11, "SUPERSET"),
    # MUTUALLY EXCLUSIVE
    (13, 14, "MUTUALLY_EXCLUSIVE"),
    # UNRELATED
    (0, 6, "UNRELATED"),
    (0, 11, "UNRELATED"),
    (6, 13, "UNRELATED"),
    (15, 0, "UNRELATED"),
]

# Map 7-label → 3-label (pipeline reduced taxonomy)
_REDUCE: dict[str, str] = {
    "IDENTICAL": "IDENTICAL",
    "COMPLEMENT": "COMPLEMENT",
    "SUBSET": "UNRELATED",
    "SUPERSET": "UNRELATED",
    "MUTUALLY_EXCLUSIVE": "UNRELATED",
    "UNRELATED": "UNRELATED",
    "AMBIGUOUS": "UNRELATED",
}


def run(use_consensus: bool = False) -> None:
    # ── Try to import pipeline classifier ─────────────────────────────────────
    pipeline_available = False
    try:
        from llm_classifier import MarketClassifier
        from models import UnifiedMarket, Exchange
        pipeline_clf = MarketClassifier()
        pipeline_available = True
        print("Pipeline classifier: AVAILABLE\n")
    except Exception as e:
        print(f"Pipeline classifier: UNAVAILABLE ({e})\n")

    total = len(PAIRS)
    correct_7 = 0
    correct_3 = 0
    pipeline_agree = 0
    pipeline_total = 0

    header = f"{'#':<3} {'EXPECTED_7':<22} {'RICH_LABEL':<22} {'CONF':>5} {'OK':<4}"
    if pipeline_available:
        header += f" {'PIPE_LABEL':<14} {'AGREE':<6}"
    print(header)
    print("─" * max(80, len(header) + 4))

    for i, (idx1, idx2, expected_7) in enumerate(PAIRS, 1):
        c1, c2 = data[idx1], data[idx2]
        pc1 = c1.get("payout_condition", str(c1))
        pc2 = c2.get("payout_condition", str(c2))

        # Rich classifier
        if use_consensus:
            result = llmd.compare_consistent(c1, c2)
        else:
            result = llmd.compare(c1, c2)

        ok7 = result.label == expected_7
        if ok7:
            correct_7 += 1
        expected_3 = _REDUCE[expected_7]
        got_3 = _REDUCE.get(result.label, "UNRELATED")
        if got_3 == expected_3:
            correct_3 += 1

        row = (
            f"{i:<3} {expected_7:<22} {result.label:<22} {result.confidence:>5.2f} "
            f"{'✅' if ok7 else '❌':<4}"
        )

        # Pipeline classifier
        if pipeline_available:
            try:
                def _stub(c: dict, idx: int) -> "UnifiedMarket":
                    q = c.get("payout_condition") or c.get("title") or f"contract_{idx}"
                    return UnifiedMarket(
                        exchange=Exchange.POLYMARKET,
                        native_id=c.get("id", f"stub_{idx}"),
                        question=q,
                        description=c.get("description", ""),
                        resolution_rules=c.get("rules", ""),
                    )

                m1 = _stub(c1, idx1)
                m2 = _stub(c2, idx2)
                pipe_result = pipeline_clf.classify_pair(m1, m2)
                pipe_label = pipe_result.label
                agree = pipe_label == expected_3
                if agree:
                    pipeline_agree += 1
                pipeline_total += 1
                row += f" {pipe_label:<14} {'✅' if agree else '❌':<6}"
            except Exception as exc:
                row += f" {'ERROR':<14} {'?':<6}"
                print(f"    [pipeline error] {exc}")
                pipeline_total += 1

        print(row)
        print(f"    A: {pc1[:90]}")
        print(f"    B: {pc2[:90]}")
        print()

    print("─" * 60)
    print(f"Rich 7-label accuracy:   {correct_7}/{total} = {correct_7/total*100:.1f}%")
    print(f"Rich→3-label accuracy:   {correct_3}/{total} = {correct_3/total*100:.1f}%")
    if pipeline_available and pipeline_total > 0:
        print(f"Pipeline 3-label agree:  {pipeline_agree}/{pipeline_total} = {pipeline_agree/pipeline_total*100:.1f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM classifier evaluation harness.")
    parser.add_argument("--consensus", action="store_true",
                        help="Use 3-vote majority consensus (slower, more accurate)")
    args = parser.parse_args()
    run(use_consensus=args.consensus)


if __name__ == "__main__":
    main()
