"""Tests for Scout._pair_exists: symmetric dedup guard prevents one-to-many pairings."""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from scout import Scout
from models import UnifiedMarket, Exchange


def _make_scout(tmp_path) -> Scout:
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text("[]")
    return Scout(str(pairs_file))


def _uid(exchange: str, native_id: str) -> str:
    return f"{exchange}:{native_id}"


def _pair_dict(uid_a: str, uid_b: str) -> dict:
    ex_a, id_a = uid_a.split(":", 1)
    ex_b, id_b = uid_b.split(":", 1)
    return {
        "pair_id": "test_pair",
        "market_a": {"uid": uid_a, "exchange": ex_a, "native_id": id_a, "question": "q"},
        "market_b": {"uid": uid_b, "exchange": ex_b, "native_id": id_b, "question": "q"},
        "relation": "IDENTICAL",
        "confidence": 0.95,
        "reason": "same",
        "discovered_at": "2026-01-01T00:00:00",
    }


# ── Exact pair already stored ──────────────────────────────────────────────────

def test_exact_pair_detected(tmp_path):
    scout = _make_scout(tmp_path)
    uid_a = _uid("kalshi", "K1")
    uid_b = _uid("polymarket", "P1")
    scout.verified_pairs.append(_pair_dict(uid_a, uid_b))

    assert scout._pair_exists(uid_a, uid_b)


def test_exact_pair_reverse_order(tmp_path):
    scout = _make_scout(tmp_path)
    uid_a = _uid("kalshi", "K1")
    uid_b = _uid("polymarket", "P1")
    scout.verified_pairs.append(_pair_dict(uid_a, uid_b))

    assert scout._pair_exists(uid_b, uid_a)


# ── Symmetric dedup: counterpart market already paired with another source ─────

def test_symmetric_block_on_uid_b(tmp_path):
    """After K1↔P1 is stored, K2↔P1 should be blocked (P1 already used)."""
    scout = _make_scout(tmp_path)
    uid_k1 = _uid("kalshi", "K1")
    uid_p1 = _uid("polymarket", "P1")
    scout.verified_pairs.append(_pair_dict(uid_k1, uid_p1))

    uid_k2 = _uid("kalshi", "K2")
    # uid_b = uid_p1 is in uids, uid_a = uid_k2 is not → should be blocked
    assert scout._pair_exists(uid_k2, uid_p1)


def test_symmetric_block_on_uid_a(tmp_path):
    """After K1↔P1 is stored, K1↔P2 should be blocked (K1 already used)."""
    scout = _make_scout(tmp_path)
    uid_k1 = _uid("kalshi", "K1")
    uid_p1 = _uid("polymarket", "P1")
    scout.verified_pairs.append(_pair_dict(uid_k1, uid_p1))

    uid_p2 = _uid("polymarket", "P2")
    # uid_a = uid_k1 is in uids, uid_b = uid_p2 is not → should be blocked
    assert scout._pair_exists(uid_k1, uid_p2)


# ── Completely unrelated markets are not blocked ───────────────────────────────

def test_unrelated_markets_not_blocked(tmp_path):
    scout = _make_scout(tmp_path)
    uid_k1 = _uid("kalshi", "K1")
    uid_p1 = _uid("polymarket", "P1")
    scout.verified_pairs.append(_pair_dict(uid_k1, uid_p1))

    uid_k2 = _uid("kalshi", "K2")
    uid_p2 = _uid("polymarket", "P2")
    # Neither K2 nor P2 appear in the stored pair → should be allowed
    assert not scout._pair_exists(uid_k2, uid_p2)
