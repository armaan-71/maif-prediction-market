import pytest

from SemanticMatchPipeline.models import (
    ArbitrageCandidate,
    ClassifiedPair,
    MarketSummary,
    RelationType,
)


@pytest.fixture
def make_market():
    def _factory(
        uid="kalshi:A",
        exchange="kalshi",
        question="Will X happen?",
        yes_price=None,
        no_price=None,
        event_id=None,
    ):
        return MarketSummary(
            uid=uid,
            question=question,
            exchange=exchange,
            yes_price=yes_price,
            no_price=no_price,
            event_id=event_id,
            embedding_text=question,
        )

    return _factory


@pytest.fixture
def make_pair(make_market):
    def _factory(a=None, b=None, relation=RelationType.IDENTICAL, confidence=0.9):
        if a is None:
            a = make_market("kalshi:A", "kalshi", yes_price=0.50)
        if b is None:
            b = make_market("poly:B", "polymarket", yes_price=0.50)
        return ClassifiedPair(
            market_a=a,
            market_b=b,
            relation=relation,
            confidence=confidence,
            differences=[],
            reason="test",
        )

    return _factory


@pytest.fixture
def make_candidate():
    def _factory(
        uid_a="kalshi:A",
        uid_b="poly:B",
        exchange_a="kalshi",
        exchange_b="polymarket",
        relation=RelationType.IDENTICAL,
        confidence=0.92,
    ):
        return ArbitrageCandidate(
            uid_a=uid_a,
            uid_b=uid_b,
            native_id_a=uid_a.split(":", 1)[1],
            native_id_b=uid_b.split(":", 1)[1],
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            question_a="Will X happen?",
            question_b="Will X occur?",
            relation=relation,
            confidence=confidence,
        )

    return _factory
