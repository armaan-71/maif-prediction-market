import json
from unittest.mock import MagicMock, patch

import pytest

from SemanticMatchPipeline.classifier import (
    _format_contract,
    _should_skip_pair,
    classify_cluster,
    classify_pair,
    has_cross_exchange_pairs,
)
from SemanticMatchPipeline.models import MarketCluster, MarketSummary, RelationType


def _market(uid, exchange, question="test question", event_id=None, yes_price=None):
    return MarketSummary(
        uid=uid, question=question, exchange=exchange,
        event_id=event_id, yes_price=yes_price, embedding_text=question,
    )


def _groq_response(relation="IDENTICAL", confidence=0.9, differences=None, reason="ok"):
    content = json.dumps({
        "relation": relation,
        "confidence": confidence,
        "differences": differences or [],
        "reason": reason,
    })
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


# ── classify_pair ─────────────────────────────────────────────────────────────

class TestClassifyPair:
    def test_returns_correct_relation(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _groq_response("COMPLEMENT", 0.88)
        result = classify_pair(_market("k:A", "kalshi"), _market("p:B", "polymarket"), client)
        assert result.relation == RelationType.COMPLEMENT
        assert result.confidence == 0.88

    def test_returns_reason_and_differences(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _groq_response(
            differences=["different exchange"], reason="same event"
        )
        result = classify_pair(_market("k:A", "kalshi"), _market("p:B", "polymarket"), client)
        assert result.differences == ["different exchange"]
        assert result.reason == "same event"

    def test_unknown_relation_falls_back_to_ambiguous(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _groq_response("NOT_A_RELATION")
        result = classify_pair(_market("k:A", "kalshi"), _market("p:B", "polymarket"), client)
        assert result.relation == RelationType.AMBIGUOUS

    def test_all_retries_fail_returns_ambiguous_with_zero_confidence(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("timeout")
        result = classify_pair(
            _market("k:A", "kalshi"), _market("p:B", "polymarket"),
            client, retries=2, backoff=0,
        )
        assert result.relation == RelationType.AMBIGUOUS
        assert result.confidence == 0.0
        assert "failed" in result.reason.lower()

    def test_retries_on_transient_failure(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("rate limit"),
            _groq_response("IDENTICAL", 0.95),
        ]
        result = classify_pair(
            _market("k:A", "kalshi"), _market("p:B", "polymarket"),
            client, retries=2, backoff=0,
        )
        assert result.relation == RelationType.IDENTICAL
        assert client.chat.completions.create.call_count == 2

    def test_market_fields_sent_to_llm(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _groq_response()
        a = _market("k:A", "kalshi", question="Will GDP exceed 3%?", yes_price=0.55)
        b = _market("p:B", "polymarket", question="Will GDP top 3%?")
        classify_pair(a, b, client)
        call_args = client.chat.completions.create.call_args
        user_content = call_args[1]["messages"][1]["content"]
        assert "Will GDP exceed 3%?" in user_content
        assert "Will GDP top 3%?" in user_content
        assert "kalshi" in user_content


# ── _should_skip_pair ─────────────────────────────────────────────────────────

class TestShouldSkipPair:
    def test_skips_same_exchange_by_default(self):
        a = _market("k:A", "kalshi")
        b = _market("k:B", "kalshi")
        assert _should_skip_pair(a, b) is True

    def test_does_not_skip_cross_exchange(self):
        a = _market("k:A", "kalshi")
        b = _market("p:B", "polymarket")
        assert _should_skip_pair(a, b) is False

    def test_skips_same_event_on_same_exchange(self):
        a = _market("k:A", "kalshi", event_id="EVT-1")
        b = _market("k:B", "kalshi", event_id="EVT-1")
        assert _should_skip_pair(a, b) is True

    def test_does_not_skip_different_events_on_same_exchange_when_cross_exchange_only_false(self):
        a = _market("k:A", "kalshi", event_id="EVT-1")
        b = _market("k:B", "kalshi", event_id="EVT-2")
        assert _should_skip_pair(a, b, cross_exchange_only=False) is False

    def test_none_event_id_does_not_match(self):
        a = _market("k:A", "kalshi", event_id=None)
        b = _market("k:B", "kalshi", event_id=None)
        # event_id is None for both — should not be treated as same event
        assert _should_skip_pair(a, b, cross_exchange_only=False) is False


# ── classify_cluster ──────────────────────────────────────────────────────────

class TestClassifyCluster:
    def test_skips_all_same_exchange_pairs(self):
        client = MagicMock()
        cluster = MarketCluster(cluster_id=0, markets=[
            _market("k:A", "kalshi"),
            _market("k:B", "kalshi"),
        ])
        result = classify_cluster(cluster, client)
        assert result == []
        client.chat.completions.create.assert_not_called()

    def test_classifies_cross_exchange_pairs(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _groq_response()
        cluster = MarketCluster(cluster_id=0, markets=[
            _market("k:A", "kalshi"),
            _market("p:B", "polymarket"),
            _market("p:C", "polymarket"),
        ])
        result = classify_cluster(cluster, client)
        # k:A × p:B and k:A × p:C are cross-exchange; p:B × p:C is same-exchange
        assert len(result) == 2

    def test_respects_max_pairs_cap(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _groq_response()
        markets = [_market(f"k:{i}" if i % 2 == 0 else f"p:{i}",
                           "kalshi" if i % 2 == 0 else "polymarket")
                   for i in range(6)]
        cluster = MarketCluster(cluster_id=0, markets=markets)
        result = classify_cluster(cluster, client, max_pairs=2)
        assert len(result) == 2

    def test_zero_max_pairs_means_no_cap(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _groq_response()
        markets = [_market(f"k:{i}" if i % 2 == 0 else f"p:{i}",
                           "kalshi" if i % 2 == 0 else "polymarket")
                   for i in range(6)]
        cluster = MarketCluster(cluster_id=0, markets=markets)
        result = classify_cluster(cluster, client, max_pairs=0)
        assert len(result) > 2


# ── has_cross_exchange_pairs ──────────────────────────────────────────────────

class TestHasCrossExchangePairs:
    def test_returns_true_for_mixed_exchanges(self):
        cluster = MarketCluster(cluster_id=0, markets=[
            _market("k:A", "kalshi"),
            _market("p:B", "polymarket"),
        ])
        assert has_cross_exchange_pairs(cluster) is True

    def test_returns_false_for_single_exchange(self):
        cluster = MarketCluster(cluster_id=0, markets=[
            _market("k:A", "kalshi"),
            _market("k:B", "kalshi"),
        ])
        assert has_cross_exchange_pairs(cluster) is False

    def test_returns_false_for_single_market(self):
        cluster = MarketCluster(cluster_id=0, markets=[_market("k:A", "kalshi")])
        assert has_cross_exchange_pairs(cluster) is False


# ── _format_contract ──────────────────────────────────────────────────────────

class TestFormatContract:
    def test_includes_required_fields(self):
        m = _market("k:A", "kalshi", question="Will X happen?", yes_price=0.55)
        text = _format_contract(m, "A")
        assert "CONTRACT A:" in text
        assert "kalshi" in text
        assert "Will X happen?" in text
        assert "0.55" in text

    def test_includes_resolution_rules_when_present(self):
        m = MarketSummary(
            uid="k:A", question="Q?", exchange="kalshi",
            resolution_rules="Resolves YES if X.", embedding_text="Q?",
        )
        text = _format_contract(m, "A")
        assert "Resolves YES if X." in text

    def test_omits_optional_fields_when_absent(self):
        m = _market("k:A", "kalshi")
        text = _format_contract(m, "A")
        assert "Resolution rules" not in text
        assert "Description" not in text
        assert "Close at" not in text
