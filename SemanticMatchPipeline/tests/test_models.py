import pytest
from pydantic import ValidationError

from SemanticMatchPipeline.models import (
    ArbPairsOutput,
    ArbitrageCandidate,
    ArbitrageOpportunity,
    ClassifiedPair,
    MarketCluster,
    MarketSummary,
    PipelineOutput,
    RelationType,
)


class TestRelationType:
    def test_all_seven_variants_exist(self):
        expected = {
            "IDENTICAL", "COMPLEMENT", "SUBSET", "SUPERSET",
            "MUTUALLY_EXCLUSIVE", "UNRELATED", "AMBIGUOUS",
        }
        assert {r.value for r in RelationType} == expected

    def test_is_string_enum(self):
        assert isinstance(RelationType.IDENTICAL, str)
        assert RelationType.IDENTICAL == "IDENTICAL"


class TestMarketSummary:
    def test_minimal_construction(self):
        m = MarketSummary(uid="kalshi:A", question="Q?", exchange="kalshi", embedding_text="Q?")
        assert m.uid == "kalshi:A"
        assert m.yes_price is None

    def test_optional_fields_default_to_none(self):
        m = MarketSummary(uid="x", question="q", exchange="e", embedding_text="q")
        assert m.event_id is None
        assert m.no_price is None
        assert m.close_at is None
        assert m.resolution_rules is None
        assert m.description is None

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            MarketSummary(uid="x", exchange="e", embedding_text="q")  # missing question


class TestClassifiedPair:
    def test_construction(self, make_market):
        a = make_market("kalshi:A", "kalshi")
        b = make_market("poly:B", "polymarket")
        pair = ClassifiedPair(
            market_a=a, market_b=b,
            relation=RelationType.IDENTICAL,
            confidence=0.95,
            differences=["different wording"],
            reason="same underlying event",
        )
        assert pair.relation == RelationType.IDENTICAL
        assert pair.confidence == 0.95
        assert pair.differences == ["different wording"]


class TestArbitrageCandidate:
    def test_round_trips_via_model_dump(self, make_candidate):
        c = make_candidate()
        data = c.model_dump()
        restored = ArbitrageCandidate.model_validate(data)
        assert restored.uid_a == c.uid_a
        assert restored.relation == c.relation

    def test_native_id_is_derived_from_uid(self, make_candidate):
        c = make_candidate(uid_a="kalshi:TICKER-X", uid_b="poly:0xabc")
        assert c.native_id_a == "TICKER-X"
        assert c.native_id_b == "0xabc"


class TestArbPairsOutput:
    def test_empty_pairs(self):
        out = ArbPairsOutput(pairs=[], generated_at="2026-01-01T00:00:00+00:00")
        assert out.pairs == []

    def test_with_candidates(self, make_candidate):
        candidates = [make_candidate(), make_candidate(uid_a="kalshi:B", uid_b="poly:C")]
        out = ArbPairsOutput(pairs=candidates, generated_at="2026-01-01T00:00:00+00:00")
        assert len(out.pairs) == 2
