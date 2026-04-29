"""
SemanticMatchPipeline orchestrator.

Usage:
    python -m SemanticMatchPipeline.pipeline
    python -m SemanticMatchPipeline.pipeline --markets UnifiedMarketPipeline/markets.json --output opportunities.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .arbitrage import MIN_ARB_EDGE, MIN_CONFIDENCE, score_pair
from .classifier import classify_cluster, has_cross_exchange_pairs, make_groq_client
from .grouper import DEFAULT_THRESHOLD, build_clusters, load_markets
from .models import (
    ArbitrageCandidate, ArbPairsOutput,
    ArbitrageOpportunity, PipelineOutput, RelationType,
)


def run(
    markets_path: str = "UnifiedMarketPipeline/markets.json",
    output_path: str = "SemanticMatchPipeline/opportunities.json",
    qdrant_path: str = "qdrant_data",
    similarity_threshold: float = DEFAULT_THRESHOLD,
    min_confidence: float = MIN_CONFIDENCE,
    min_arb_edge: float = MIN_ARB_EDGE,
    max_pairs_per_cluster: int = 100,
) -> PipelineOutput:
    # ── 1. Load markets ──────────────────────────────────────────────────────
    print(f"[pipeline] Loading markets from {markets_path}...")
    markets = load_markets(markets_path)
    print(f"[pipeline] Loaded {len(markets)} markets.")

    # ── 2. Cluster via Qdrant KNN ────────────────────────────────────────────
    print(f"[pipeline] Building clusters (threshold={similarity_threshold})...")
    clusters = build_clusters(markets, qdrant_path=qdrant_path, threshold=similarity_threshold)
    print(f"[pipeline] Found {len(clusters)} clusters with 2+ markets.")

    if not clusters:
        print("[pipeline] No clusters found. Try lowering --threshold or re-running the vector pipeline.")
        return PipelineOutput(
            opportunities=[],
            unrelated_pairs=0,
            ambiguous_pairs=0,
            total_clusters=0,
            total_pairs_classified=0,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── 3. LLM classification ────────────────────────────────────────────────
    # Pre-filter: skip clusters with no cross-exchange pairs to avoid wasting tokens.
    cross_clusters = [c for c in clusters if has_cross_exchange_pairs(c)]
    skipped = len(clusters) - len(cross_clusters)
    print(
        f"[pipeline] {len(cross_clusters)} clusters have cross-exchange pairs "
        f"({skipped} single-exchange clusters skipped)."
    )

    groq = make_groq_client()
    all_pairs = []
    for cluster in cross_clusters:
        n = len(cluster.markets)
        cross_count = sum(
            1 for i, a in enumerate(cluster.markets)
            for b in cluster.markets[i + 1:]
            if a.exchange != b.exchange
        )
        capped = f" → capping at {max_pairs_per_cluster}" if cross_count > max_pairs_per_cluster else ""
        print(
            f"[pipeline] Cluster {cluster.cluster_id}: {n} markets, "
            f"{cross_count} cross-exchange pairs{capped}"
        )
        pairs = classify_cluster(cluster, groq, max_pairs=max_pairs_per_cluster)
        all_pairs.extend(pairs)

    print(f"[pipeline] Classified {len(all_pairs)} pairs total.")

    # ── 4. Build price-free arb candidates (written to arb_pairs.json) ───────
    # Only keep pairs with actionable relations and sufficient confidence.
    # Prices are intentionally excluded here — the poller fetches live quotes.
    now = datetime.now(timezone.utc).isoformat()
    candidates: list[ArbitrageCandidate] = []
    for pair in all_pairs:
        if pair.relation in (RelationType.UNRELATED, RelationType.AMBIGUOUS):
            continue
        if pair.confidence < min_confidence:
            continue
        a, b = pair.market_a, pair.market_b
        candidates.append(ArbitrageCandidate(
            uid_a=a.uid,
            uid_b=b.uid,
            native_id_a=a.uid.split(":", 1)[1],
            native_id_b=b.uid.split(":", 1)[1],
            exchange_a=a.exchange,
            exchange_b=b.exchange,
            question_a=a.question,
            question_b=b.question,
            relation=pair.relation,
            confidence=pair.confidence,
        ))

    pairs_path = Path(output_path).parent / "arb_pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(ArbPairsOutput(pairs=candidates, generated_at=now).model_dump(), f, indent=2, default=str)
    print(f"[pipeline] Wrote {len(candidates)} arb candidates to {pairs_path}")

    # ── 5. Score against snapshot prices (for debugging / last-run summary) ──
    opportunities: list[ArbitrageOpportunity] = []
    unrelated = 0
    ambiguous = 0

    for pair in all_pairs:
        if pair.relation == RelationType.UNRELATED:
            unrelated += 1
        elif pair.relation == RelationType.AMBIGUOUS:
            ambiguous += 1

        opp = score_pair(pair, min_confidence=min_confidence, min_arb_edge=min_arb_edge)
        if opp:
            opportunities.append(opp)

    opportunities.sort(key=lambda o: o.arb_edge, reverse=True)
    print(f"[pipeline] Found {len(opportunities)} snapshot opportunities (prices may be stale).")

    # ── 6. Write snapshot output ─────────────────────────────────────────────
    result = PipelineOutput(
        opportunities=opportunities,
        unrelated_pairs=unrelated,
        ambiguous_pairs=ambiguous,
        total_clusters=len(clusters),
        total_pairs_classified=len(all_pairs),
        generated_at=now,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)
    print(f"[pipeline] Wrote snapshot to {output_path}")

    _print_summary(result)
    return result


def _print_summary(result: PipelineOutput) -> None:
    print("\n" + "=" * 72)
    print("ARBITRAGE OPPORTUNITIES")
    print("=" * 72)
    if not result.opportunities:
        print("  No opportunities detected.")
    for opp in result.opportunities:
        a = opp.pair.market_a
        b = opp.pair.market_b
        print(
            f"\n  Edge: {opp.arb_edge:.4f}  |  Relation: {opp.pair.relation.value}"
            f"  |  Confidence: {opp.pair.confidence:.2f}"
        )
        print(f"  A [{a.exchange}] {a.question[:70]}")
        print(f"  B [{b.exchange}] {b.question[:70]}")
        print(f"  Strategy: {opp.strategy}")

    print("\n" + "-" * 72)
    print(
        f"  Clusters: {result.total_clusters}  |  "
        f"Pairs classified: {result.total_pairs_classified}  |  "
        f"Unrelated: {result.unrelated_pairs}  |  "
        f"Ambiguous: {result.ambiguous_pairs}"
    )
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SemanticMatchPipeline")
    parser.add_argument(
        "--markets",
        default="UnifiedMarketPipeline/markets.json",
        help="Path to the unified markets JSON file",
    )
    parser.add_argument(
        "--output",
        default="SemanticMatchPipeline/opportunities.json",
        help="Path to write the opportunities JSON",
    )
    parser.add_argument(
        "--qdrant",
        default="qdrant_data",
        help="Path to local Qdrant data directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity score to link two markets in a cluster",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help="Minimum LLM confidence to consider a pair",
    )
    parser.add_argument(
        "--min-arb-edge",
        type=float,
        default=MIN_ARB_EDGE,
        help="Minimum arb edge (profit per $1) to include in output",
    )
    parser.add_argument(
        "--max-pairs-per-cluster",
        type=int,
        default=100,
        help="Cap LLM calls per cluster to conserve API quota (0 = no cap)",
    )
    args = parser.parse_args()

    run(
        markets_path=args.markets,
        output_path=args.output,
        qdrant_path=args.qdrant,
        similarity_threshold=args.threshold,
        min_confidence=args.min_confidence,
        min_arb_edge=args.min_arb_edge,
        max_pairs_per_cluster=args.max_pairs_per_cluster,
    )


if __name__ == "__main__":
    main()
