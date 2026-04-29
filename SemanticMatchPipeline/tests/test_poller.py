import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from SemanticMatchPipeline.poller import (
    _candidate_to_pair,
    _kalshi_price,
    _manifold_price,
    _polymarket_price,
    fetch_live_price,
    poll_once,
)
from SemanticMatchPipeline.models import ArbitrageCandidate, RelationType


def _mock_http(status_code, json_body):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    client = AsyncMock()
    client.get.return_value = resp
    return client


# ── _kalshi_price ─────────────────────────────────────────────────────────────

class TestKalshiPrice:
    def test_parses_dollars_fields(self):
        client = _mock_http(200, {"market": {"yes_bid_dollars": "0.56", "no_bid_dollars": "0.44"}})
        yes, no = asyncio.run(_kalshi_price("TICKER", client))
        assert yes == pytest.approx(0.56)
        assert no == pytest.approx(0.44)

    def test_falls_back_to_cents_fields(self):
        client = _mock_http(200, {"market": {"yes_bid": 60, "no_bid": 40}})
        yes, no = asyncio.run(_kalshi_price("TICKER", client))
        assert yes == pytest.approx(0.60)
        assert no == pytest.approx(0.40)

    def test_dollars_takes_precedence_over_cents(self):
        client = _mock_http(200, {"market": {"yes_bid_dollars": "0.55", "yes_bid": 70}})
        yes, _ = asyncio.run(_kalshi_price("TICKER", client))
        assert yes == pytest.approx(0.55)

    def test_returns_none_on_non_200(self):
        client = _mock_http(404, {})
        yes, no = asyncio.run(_kalshi_price("NONEXISTENT", client))
        assert yes is None and no is None

    def test_returns_none_for_zero_price(self):
        # Zero means no bids posted — should be treated as unavailable
        client = _mock_http(200, {"market": {"yes_bid_dollars": "0.00"}})
        yes, _ = asyncio.run(_kalshi_price("TICKER", client))
        assert yes is None


# ── _polymarket_price ─────────────────────────────────────────────────────────

class TestPolymarketPrice:
    def test_parses_outcome_prices_string(self):
        client = _mock_http(200, [{"outcomePrices": '["0.65", "0.35"]'}])
        yes, no = asyncio.run(_polymarket_price("0xabc", client))
        assert yes == pytest.approx(0.65)
        assert no == pytest.approx(0.35)

    def test_parses_outcome_prices_list(self):
        client = _mock_http(200, [{"outcomePrices": [0.70, 0.30]}])
        yes, no = asyncio.run(_polymarket_price("0xabc", client))
        assert yes == pytest.approx(0.70)
        assert no == pytest.approx(0.30)

    def test_returns_none_on_non_200(self):
        client = _mock_http(404, {})
        yes, no = asyncio.run(_polymarket_price("0xabc", client))
        assert yes is None and no is None

    def test_returns_none_for_empty_response(self):
        client = _mock_http(200, [])
        yes, no = asyncio.run(_polymarket_price("0xabc", client))
        assert yes is None and no is None


# ── _manifold_price ───────────────────────────────────────────────────────────

class TestManifoldPrice:
    def test_parses_probability(self):
        client = _mock_http(200, {"probability": 0.72})
        yes, no = asyncio.run(_manifold_price("abc123", client))
        assert yes == pytest.approx(0.72)
        assert no == pytest.approx(0.28)

    def test_returns_none_when_probability_missing(self):
        client = _mock_http(200, {"title": "no prob field"})
        yes, no = asyncio.run(_manifold_price("abc123", client))
        assert yes is None and no is None

    def test_returns_none_on_non_200(self):
        client = _mock_http(404, {})
        yes, no = asyncio.run(_manifold_price("abc123", client))
        assert yes is None and no is None


# ── fetch_live_price ──────────────────────────────────────────────────────────

class TestFetchLivePrice:
    def test_routes_to_kalshi(self):
        client = _mock_http(200, {"market": {"yes_bid_dollars": "0.60"}})
        yes, _ = asyncio.run(fetch_live_price("kalshi", "TICKER", client))
        assert yes == pytest.approx(0.60)

    def test_routes_to_polymarket(self):
        client = _mock_http(200, [{"outcomePrices": '["0.45", "0.55"]'}])
        yes, _ = asyncio.run(fetch_live_price("polymarket", "0xabc", client))
        assert yes == pytest.approx(0.45)

    def test_routes_to_manifold(self):
        client = _mock_http(200, {"probability": 0.33})
        yes, _ = asyncio.run(fetch_live_price("manifold", "slug", client))
        assert yes == pytest.approx(0.33)

    def test_unknown_exchange_returns_none(self):
        client = AsyncMock()
        yes, no = asyncio.run(fetch_live_price("unknown_exchange", "id", client))
        assert yes is None and no is None
        client.get.assert_not_called()

    def test_exception_returns_none(self):
        client = AsyncMock()
        client.get.side_effect = Exception("network error")
        yes, no = asyncio.run(fetch_live_price("kalshi", "TICKER", client))
        assert yes is None and no is None


# ── _candidate_to_pair ────────────────────────────────────────────────────────

class TestCandidateToPair:
    def test_assembles_classified_pair(self, make_candidate):
        c = make_candidate()
        pair = _candidate_to_pair(c, 0.40, 0.60, 0.60, 0.40)
        assert pair.market_a.uid == c.uid_a
        assert pair.market_b.uid == c.uid_b
        assert pair.market_a.yes_price == 0.40
        assert pair.market_b.yes_price == 0.60
        assert pair.relation == c.relation
        assert pair.confidence == c.confidence

    def test_preserves_none_prices(self, make_candidate):
        c = make_candidate()
        pair = _candidate_to_pair(c, None, None, 0.60, 0.40)
        assert pair.market_a.yes_price is None


# ── poll_once ─────────────────────────────────────────────────────────────────

class TestPollOnce:
    def test_returns_empty_when_file_missing(self):
        result = asyncio.run(poll_once("/nonexistent/arb_pairs.json"))
        assert result == []

    def test_returns_empty_when_no_candidates(self, tmp_path):
        pairs_file = tmp_path / "arb_pairs.json"
        pairs_file.write_text(json.dumps({"pairs": [], "generated_at": "2026-01-01T00:00:00+00:00"}))
        result = asyncio.run(poll_once(str(pairs_file)))
        assert result == []

    def test_detects_opportunity_above_threshold(self, tmp_path, make_candidate):
        c = make_candidate(relation=RelationType.IDENTICAL, confidence=0.92)
        pairs_file = tmp_path / "arb_pairs.json"
        pairs_file.write_text(json.dumps({
            "pairs": [c.model_dump()],
            "generated_at": "2026-01-01T00:00:00+00:00",
        }))

        with patch("SemanticMatchPipeline.poller.fetch_live_price", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [(0.40, 0.60), (0.60, 0.40)]
            found = asyncio.run(poll_once(str(pairs_file), min_arb_edge=0.02))

        assert len(found) == 1
        assert found[0]["edge"] == pytest.approx(0.20, abs=1e-4)
        assert found[0]["uid_a"] == c.uid_a
        assert found[0]["uid_b"] == c.uid_b

    def test_skips_pair_when_price_unavailable(self, tmp_path, make_candidate):
        c = make_candidate()
        pairs_file = tmp_path / "arb_pairs.json"
        pairs_file.write_text(json.dumps({
            "pairs": [c.model_dump()],
            "generated_at": "2026-01-01T00:00:00+00:00",
        }))

        with patch("SemanticMatchPipeline.poller.fetch_live_price", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [(None, None), (0.60, 0.40)]
            found = asyncio.run(poll_once(str(pairs_file)))

        assert found == []

    def test_returns_empty_when_edge_below_threshold(self, tmp_path, make_candidate):
        c = make_candidate(relation=RelationType.IDENTICAL)
        pairs_file = tmp_path / "arb_pairs.json"
        pairs_file.write_text(json.dumps({
            "pairs": [c.model_dump()],
            "generated_at": "2026-01-01T00:00:00+00:00",
        }))

        with patch("SemanticMatchPipeline.poller.fetch_live_price", new_callable=AsyncMock) as mock_fetch:
            # Edge = 0.01, below default min_arb_edge of 0.02
            mock_fetch.side_effect = [(0.50, 0.50), (0.51, 0.49)]
            found = asyncio.run(poll_once(str(pairs_file), min_arb_edge=0.02))

        assert found == []

    def test_opportunity_record_has_expected_keys(self, tmp_path, make_candidate):
        c = make_candidate(relation=RelationType.IDENTICAL, confidence=0.92)
        pairs_file = tmp_path / "arb_pairs.json"
        pairs_file.write_text(json.dumps({
            "pairs": [c.model_dump()],
            "generated_at": "2026-01-01T00:00:00+00:00",
        }))

        with patch("SemanticMatchPipeline.poller.fetch_live_price", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [(0.40, 0.60), (0.60, 0.40)]
            found = asyncio.run(poll_once(str(pairs_file)))

        record = found[0]
        for key in ("edge", "relation", "confidence", "strategy", "uid_a", "uid_b",
                    "question_a", "question_b", "yes_price_a", "yes_price_b", "checked_at"):
            assert key in record, f"missing key: {key}"
