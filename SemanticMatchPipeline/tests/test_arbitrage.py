import pytest

from SemanticMatchPipeline.arbitrage import MIN_ARB_EDGE, MIN_CONFIDENCE, score_pair
from SemanticMatchPipeline.models import ClassifiedPair, MarketSummary, RelationType


def _market(uid, exchange, yes_price):
    return MarketSummary(
        uid=uid, question="test", exchange=exchange,
        yes_price=yes_price, embedding_text="test",
    )


def _pair(a, b, relation, confidence=0.9):
    return ClassifiedPair(
        market_a=a, market_b=b,
        relation=relation, confidence=confidence,
        differences=[], reason="",
    )


# ── IDENTICAL ─────────────────────────────────────────────────────────────────

class TestIdentical:
    def test_arb_when_prices_differ(self):
        pair = _pair(_market("k:A", "kalshi", 0.40), _market("p:B", "polymarket", 0.60), RelationType.IDENTICAL)
        opp = score_pair(pair)
        assert opp is not None
        assert opp.arb_edge == pytest.approx(0.20, abs=1e-4)

    def test_strategy_names_cheaper_exchange_first(self):
        pair = _pair(_market("k:A", "kalshi", 0.40), _market("p:B", "polymarket", 0.60), RelationType.IDENTICAL)
        opp = score_pair(pair)
        assert "kalshi" in opp.strategy
        assert "polymarket" in opp.strategy
        assert opp.strategy.index("kalshi") < opp.strategy.index("polymarket")

    def test_no_arb_when_prices_equal(self):
        pair = _pair(_market("k:A", "kalshi", 0.50), _market("p:B", "polymarket", 0.50), RelationType.IDENTICAL)
        assert score_pair(pair) is None

    def test_edge_is_rounded_to_four_decimals(self):
        pair = _pair(_market("k:A", "kalshi", 0.3333), _market("p:B", "polymarket", 0.5556), RelationType.IDENTICAL)
        opp = score_pair(pair)
        assert opp is not None
        assert opp.arb_edge == round(abs(0.3333 - 0.5556), 4)


# ── COMPLEMENT ────────────────────────────────────────────────────────────────

class TestComplement:
    def test_arb_when_sum_below_one(self):
        # Buy both YES for 0.70 total → guaranteed $1 payout
        pair = _pair(_market("k:A", "kalshi", 0.30), _market("p:B", "polymarket", 0.40), RelationType.COMPLEMENT)
        opp = score_pair(pair)
        assert opp is not None
        assert opp.arb_edge == pytest.approx(0.30, abs=1e-4)
        assert "YES_A" in opp.strategy and "YES_B" in opp.strategy

    def test_arb_when_sum_above_one(self):
        # Buy both NO for 0.70 total → guaranteed $1 payout
        pair = _pair(_market("k:A", "kalshi", 0.70), _market("p:B", "polymarket", 0.60), RelationType.COMPLEMENT)
        opp = score_pair(pair)
        assert opp is not None
        assert opp.arb_edge == pytest.approx(0.30, abs=1e-4)
        assert "NO_A" in opp.strategy and "NO_B" in opp.strategy

    def test_no_arb_when_sum_equals_one(self):
        pair = _pair(_market("k:A", "kalshi", 0.50), _market("p:B", "polymarket", 0.50), RelationType.COMPLEMENT)
        assert score_pair(pair) is None

    def test_no_prices_in_strategy_string_are_correct(self):
        pair = _pair(_market("k:A", "kalshi", 0.70), _market("p:B", "polymarket", 0.60), RelationType.COMPLEMENT)
        opp = score_pair(pair)
        # NO_A price should be 1 - 0.70 = 0.300, NO_B = 1 - 0.60 = 0.400
        assert "0.300" in opp.strategy
        assert "0.400" in opp.strategy


# ── SUBSET ────────────────────────────────────────────────────────────────────

class TestSubset:
    def test_arb_when_pa_above_pb(self):
        # A ⊂ B means P(A) must be ≤ P(B); if it's not, sell A buy B
        pair = _pair(_market("k:A", "kalshi", 0.60), _market("p:B", "polymarket", 0.40), RelationType.SUBSET)
        opp = score_pair(pair)
        assert opp is not None
        assert opp.arb_edge == pytest.approx(0.20, abs=1e-4)
        assert "Sell YES_A" in opp.strategy

    def test_no_arb_when_pa_below_pb(self):
        pair = _pair(_market("k:A", "kalshi", 0.30), _market("p:B", "polymarket", 0.60), RelationType.SUBSET)
        assert score_pair(pair) is None

    def test_no_arb_when_pa_equals_pb(self):
        pair = _pair(_market("k:A", "kalshi", 0.50), _market("p:B", "polymarket", 0.50), RelationType.SUBSET)
        assert score_pair(pair) is None


# ── SUPERSET ──────────────────────────────────────────────────────────────────

class TestSuperset:
    def test_arb_when_pa_below_pb(self):
        # A ⊃ B means P(A) must be ≥ P(B); if it's not, buy A sell B
        pair = _pair(_market("k:A", "kalshi", 0.30), _market("p:B", "polymarket", 0.60), RelationType.SUPERSET)
        opp = score_pair(pair)
        assert opp is not None
        assert opp.arb_edge == pytest.approx(0.30, abs=1e-4)
        assert "Buy YES_A" in opp.strategy

    def test_no_arb_when_pa_above_pb(self):
        pair = _pair(_market("k:A", "kalshi", 0.70), _market("p:B", "polymarket", 0.40), RelationType.SUPERSET)
        assert score_pair(pair) is None


# ── MUTUALLY_EXCLUSIVE ────────────────────────────────────────────────────────

class TestMutuallyExclusive:
    def test_arb_when_sum_above_one(self):
        # P(A) + P(B) must be ≤ 1; if sum > 1, sell both YES
        pair = _pair(_market("k:A", "kalshi", 0.60), _market("p:B", "polymarket", 0.50), RelationType.MUTUALLY_EXCLUSIVE)
        opp = score_pair(pair)
        assert opp is not None
        assert opp.arb_edge == pytest.approx(0.10, abs=1e-4)
        assert "mutually exclusive" in opp.strategy

    def test_no_arb_when_sum_equals_one(self):
        pair = _pair(_market("k:A", "kalshi", 0.50), _market("p:B", "polymarket", 0.50), RelationType.MUTUALLY_EXCLUSIVE)
        assert score_pair(pair) is None

    def test_no_arb_when_sum_below_one(self):
        pair = _pair(_market("k:A", "kalshi", 0.30), _market("p:B", "polymarket", 0.40), RelationType.MUTUALLY_EXCLUSIVE)
        assert score_pair(pair) is None


# ── Guards ────────────────────────────────────────────────────────────────────

class TestGuards:
    def test_low_confidence_returns_none(self):
        pair = _pair(
            _market("k:A", "kalshi", 0.40), _market("p:B", "polymarket", 0.60),
            RelationType.IDENTICAL, confidence=MIN_CONFIDENCE - 0.01,
        )
        assert score_pair(pair) is None

    def test_confidence_at_threshold_is_accepted(self):
        pair = _pair(
            _market("k:A", "kalshi", 0.40), _market("p:B", "polymarket", 0.60),
            RelationType.IDENTICAL, confidence=MIN_CONFIDENCE,
        )
        assert score_pair(pair) is not None

    def test_unrelated_returns_none(self):
        pair = _pair(_market("k:A", "kalshi", 0.40), _market("p:B", "polymarket", 0.60), RelationType.UNRELATED)
        assert score_pair(pair) is None

    def test_ambiguous_returns_none(self):
        pair = _pair(_market("k:A", "kalshi", 0.40), _market("p:B", "polymarket", 0.60), RelationType.AMBIGUOUS)
        assert score_pair(pair) is None

    def test_missing_yes_price_returns_none(self):
        pair = _pair(_market("k:A", "kalshi", None), _market("p:B", "polymarket", 0.60), RelationType.IDENTICAL)
        assert score_pair(pair) is None

    def test_edge_below_min_returns_none(self):
        # Edge of 0.01 is below MIN_ARB_EDGE (0.02)
        pair = _pair(_market("k:A", "kalshi", 0.50), _market("p:B", "polymarket", 0.51), RelationType.IDENTICAL)
        assert score_pair(pair) is None

    def test_edge_at_min_is_accepted(self):
        pair = _pair(
            _market("k:A", "kalshi", 0.50),
            _market("p:B", "polymarket", 0.50 + MIN_ARB_EDGE),
            RelationType.IDENTICAL,
        )
        assert score_pair(pair) is not None

    def test_custom_thresholds_respected(self):
        pair = _pair(_market("k:A", "kalshi", 0.40), _market("p:B", "polymarket", 0.60), RelationType.IDENTICAL)
        assert score_pair(pair, min_arb_edge=0.25) is None
        assert score_pair(pair, min_arb_edge=0.10) is not None
