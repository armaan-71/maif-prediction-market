"""Tests for PriceTick JSONL serialization and HistoricalDataCache in-memory optimization."""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from historical_data import PriceTick, HistoricalDataCache


# ── PriceTick roundtrip ────────────────────────────────────────────────────────

def test_price_tick_roundtrip_full():
    original = PriceTick(
        timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        yes_price=0.65,
        no_price=0.35,
        volume=1000.0,
    )
    restored = PriceTick.from_jsonl(original.to_jsonl())

    assert abs(restored.timestamp.timestamp() - original.timestamp.timestamp()) < 1e-3
    assert abs(restored.yes_price - 0.65) < 1e-9
    assert abs(restored.no_price - 0.35) < 1e-9
    assert abs(restored.volume - 1000.0) < 1e-9


def test_price_tick_roundtrip_optional_none():
    """no_price and volume can be None — should survive the roundtrip."""
    original = PriceTick(
        timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        yes_price=0.50,
    )
    restored = PriceTick.from_jsonl(original.to_jsonl())

    assert restored.no_price is None
    assert restored.volume is None


def test_price_tick_preserves_timezone():
    original = PriceTick(
        timestamp=datetime(2026, 5, 1, 8, 30, 0, tzinfo=timezone.utc),
        yes_price=0.72,
    )
    restored = PriceTick.from_jsonl(original.to_jsonl())
    # from_jsonl always returns UTC-aware datetimes
    assert restored.timestamp.tzinfo is not None


# ── HistoricalDataCache: latest_timestamp optimization ────────────────────────

def test_cache_latest_timestamp_in_memory(tmp_path):
    cache = HistoricalDataCache(str(tmp_path))
    t1 = PriceTick(timestamp=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc), yes_price=0.5)
    t2 = PriceTick(timestamp=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc), yes_price=0.6)

    cache.append("kalshi", "K1", [t1, t2])

    latest = cache.latest_timestamp("kalshi", "K1")
    assert latest is not None
    assert abs(latest.timestamp() - t2.timestamp.timestamp()) < 1e-3


def test_cache_latest_timestamp_none_when_empty(tmp_path):
    cache = HistoricalDataCache(str(tmp_path))
    assert cache.latest_timestamp("kalshi", "MISSING") is None


def test_cache_append_multiple_batches(tmp_path):
    cache = HistoricalDataCache(str(tmp_path))
    t1 = PriceTick(timestamp=datetime(2026, 5, 1, 10, tzinfo=timezone.utc), yes_price=0.4)
    t2 = PriceTick(timestamp=datetime(2026, 5, 2, 10, tzinfo=timezone.utc), yes_price=0.5)

    cache.append("kalshi", "K1", [t1])
    cache.append("kalshi", "K1", [t2])

    # Read back from disk to confirm durability
    cache2 = HistoricalDataCache(str(tmp_path))
    ticks = cache2.load("kalshi", "K1")
    assert len(ticks) == 2
    assert abs(ticks[0].yes_price - 0.4) < 1e-9
    assert abs(ticks[1].yes_price - 0.5) < 1e-9


def test_cache_latest_reflects_second_append(tmp_path):
    cache = HistoricalDataCache(str(tmp_path))
    t1 = PriceTick(timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc), yes_price=0.3)
    t2 = PriceTick(timestamp=datetime(2026, 5, 3, tzinfo=timezone.utc), yes_price=0.7)

    cache.append("kalshi", "K1", [t1])
    latest_after_first = cache.latest_timestamp("kalshi", "K1")

    cache.append("kalshi", "K1", [t2])
    latest_after_second = cache.latest_timestamp("kalshi", "K1")

    assert latest_after_second > latest_after_first


# ── HistoricalDataCache: data isolation between markets ───────────────────────

def test_cache_separate_markets_isolated(tmp_path):
    cache = HistoricalDataCache(str(tmp_path))
    t_k = PriceTick(timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc), yes_price=0.3)
    t_p = PriceTick(timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc), yes_price=0.7)

    cache.append("kalshi", "K1", [t_k])
    cache.append("polymarket", "P1", [t_p])

    assert len(cache.load("kalshi", "K1")) == 1
    assert len(cache.load("polymarket", "P1")) == 1
    assert abs(cache.load("kalshi", "K1")[0].yes_price - 0.3) < 1e-9
    assert abs(cache.load("polymarket", "P1")[0].yes_price - 0.7) < 1e-9
