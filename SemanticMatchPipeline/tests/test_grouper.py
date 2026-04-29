import json
from unittest.mock import MagicMock, patch

import pytest

from SemanticMatchPipeline.grouper import _UnionFind, build_clusters, load_markets


# ── _UnionFind ────────────────────────────────────────────────────────────────

class TestUnionFind:
    def test_each_key_starts_in_own_group(self):
        uf = _UnionFind(["a", "b", "c"])
        assert uf.find("a") != uf.find("b")
        assert uf.find("b") != uf.find("c")

    def test_union_merges_two_groups(self):
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")
        assert uf.find("c") != uf.find("a")

    def test_union_is_transitive(self):
        uf = _UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("b") == uf.find("c")
        assert uf.find("d") != uf.find("a")

    def test_groups_returns_all_members(self):
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        groups = uf.groups()
        all_members = {m for members in groups.values() for m in members}
        assert all_members == {"a", "b", "c"}

    def test_groups_merges_connected_components(self):
        uf = _UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("c", "d")
        groups = {frozenset(v) for v in uf.groups().values()}
        assert frozenset({"a", "b"}) in groups
        assert frozenset({"c", "d"}) in groups

    def test_union_is_idempotent(self):
        uf = _UnionFind(["a", "b"])
        uf.union("a", "b")
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")


# ── load_markets ──────────────────────────────────────────────────────────────

class TestLoadMarkets:
    def test_reads_valid_json(self, tmp_path):
        data = [{"uid": "kalshi:A", "question": "Q?", "exchange": "kalshi"}]
        f = tmp_path / "markets.json"
        f.write_text(json.dumps(data))
        assert load_markets(str(f)) == data

    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_markets("/nonexistent/markets.json")

    def test_reads_multiple_markets(self, tmp_path):
        data = [
            {"uid": "kalshi:A", "question": "Q1?", "exchange": "kalshi"},
            {"uid": "poly:B", "question": "Q2?", "exchange": "polymarket"},
        ]
        f = tmp_path / "markets.json"
        f.write_text(json.dumps(data))
        result = load_markets(str(f))
        assert len(result) == 2
        assert result[1]["uid"] == "poly:B"


# ── build_clusters ────────────────────────────────────────────────────────────

def _make_qdrant_point(uid, score):
    point = MagicMock()
    point.score = score
    point.payload = {"uid": uid}
    return point


@patch("SemanticMatchPipeline.grouper._embed")
@patch("SemanticMatchPipeline.grouper._setup_client")
class TestBuildClusters:
    def test_groups_two_similar_markets(self, mock_setup, mock_embed):
        mock_embed.return_value = [[0.1, 0.2], [0.1, 0.2]]

        mock_client = MagicMock()
        mock_setup.return_value = mock_client

        def _query(collection_name, query, using, limit, with_payload):
            result = MagicMock()
            # Every market is a high-similarity neighbor of the other
            result.points = [
                _make_qdrant_point("kalshi:A", 0.95),
                _make_qdrant_point("poly:B", 0.92),
            ]
            return result

        mock_client.query_points.side_effect = _query

        markets = [
            {"uid": "kalshi:A", "question": "Will X happen?", "exchange": "kalshi"},
            {"uid": "poly:B", "question": "Will X occur?", "exchange": "polymarket"},
        ]
        clusters = build_clusters(markets, qdrant_path="fake")

        assert len(clusters) == 1
        assert {m.uid for m in clusters[0].markets} == {"kalshi:A", "poly:B"}

    def test_returns_empty_when_collection_missing(self, mock_setup, mock_embed):
        mock_client = MagicMock()
        mock_setup.return_value = mock_client
        mock_client.get_collection.side_effect = ValueError("not found")

        result = build_clusters(
            [{"uid": "kalshi:A", "question": "Q", "exchange": "kalshi"}],
            qdrant_path="fake",
        )
        assert result == []

    def test_skips_markets_below_similarity_threshold(self, mock_setup, mock_embed):
        mock_embed.return_value = [[0.1, 0.2], [0.9, 0.8]]

        mock_client = MagicMock()
        mock_setup.return_value = mock_client

        def _query(collection_name, query, using, limit, with_payload):
            result = MagicMock()
            # Score is below the default threshold of 0.78
            result.points = [_make_qdrant_point("poly:B", 0.50)]
            return result

        mock_client.query_points.side_effect = _query

        markets = [
            {"uid": "kalshi:A", "question": "Q1", "exchange": "kalshi"},
            {"uid": "poly:B", "question": "Q2", "exchange": "polymarket"},
        ]
        clusters = build_clusters(markets, qdrant_path="fake", threshold=0.78)
        assert clusters == []

    def test_skips_isolated_markets(self, mock_setup, mock_embed):
        mock_embed.return_value = [[0.1, 0.2]]

        mock_client = MagicMock()
        mock_setup.return_value = mock_client

        def _query(collection_name, query, using, limit, with_payload):
            result = MagicMock()
            result.points = []
            return result

        mock_client.query_points.side_effect = _query

        markets = [{"uid": "kalshi:A", "question": "Q", "exchange": "kalshi"}]
        clusters = build_clusters(markets, qdrant_path="fake")
        assert clusters == []

    def test_excludes_self_from_neighbors(self, mock_setup, mock_embed):
        mock_embed.return_value = [[0.1, 0.2]]

        mock_client = MagicMock()
        mock_setup.return_value = mock_client

        def _query(collection_name, query, using, limit, with_payload):
            result = MagicMock()
            # Only neighbor returned is the market itself
            result.points = [_make_qdrant_point("kalshi:A", 1.0)]
            return result

        mock_client.query_points.side_effect = _query

        markets = [{"uid": "kalshi:A", "question": "Q", "exchange": "kalshi"}]
        clusters = build_clusters(markets, qdrant_path="fake")
        assert clusters == []

    def test_skips_markets_without_uid(self, mock_setup, mock_embed):
        mock_embed.return_value = [[0.1, 0.2]]

        mock_client = MagicMock()
        mock_setup.return_value = mock_client

        def _query(collection_name, query, using, limit, with_payload):
            result = MagicMock()
            result.points = []
            return result

        mock_client.query_points.side_effect = _query

        markets = [
            {"question": "Q", "exchange": "kalshi"},  # no uid
            {"uid": "poly:B", "question": "Q2", "exchange": "polymarket"},
        ]
        clusters = build_clusters(markets, qdrant_path="fake")
        assert clusters == []

    def test_three_way_cluster_via_transitivity(self, mock_setup, mock_embed):
        mock_embed.return_value = [[0.1], [0.1], [0.1]]

        mock_client = MagicMock()
        mock_setup.return_value = mock_client

        # A links to B, B links to C — all three should end up in one cluster
        neighbor_map = {
            "kalshi:A": [_make_qdrant_point("poly:B", 0.95)],
            "poly:B":   [_make_qdrant_point("manifold:C", 0.90)],
            "manifold:C": [],
        }

        def _query(collection_name, query, using, limit, with_payload):
            result = MagicMock()
            # Identify which market we're querying by call order
            call_count = mock_client.query_points.call_count
            uids = ["kalshi:A", "poly:B", "manifold:C"]
            result.points = neighbor_map[uids[call_count - 1]]
            return result

        mock_client.query_points.side_effect = _query

        markets = [
            {"uid": "kalshi:A", "question": "Q1", "exchange": "kalshi"},
            {"uid": "poly:B", "question": "Q2", "exchange": "polymarket"},
            {"uid": "manifold:C", "question": "Q3", "exchange": "manifold"},
        ]
        clusters = build_clusters(markets, qdrant_path="fake")
        assert len(clusters) == 1
        assert len(clusters[0].markets) == 3
