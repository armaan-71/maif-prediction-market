

def get_test_data():
    return [

        # ===== BTC 110k CLUSTER (IDENTICAL + COMPLEMENT) =====
        {
            "id": "btc_110k_a",
            "event_type": "price_threshold",
            "underlying": "BTC",
            "operator": ">=",
            "threshold": 110000,
            "deadline": "2026-06-30T23:59:59Z",
            "timezone": "UTC",
            "source": "Chainlink BTC/USD",
            "payout_condition": "YES iff max spot price >= threshold before deadline"
        },
        {
            "id": "btc_110k_b",  # IDENTICAL to a
            "event_type": "price_threshold",
            "underlying": "BTC",
            "operator": ">=",
            "threshold": 110000,
            "deadline": "2026-07-01T00:00:00Z",
            "timezone": "UTC",
            "source": "Chainlink BTC/USD",
            "payout_condition": "YES iff BTC trades at or above threshold before July 1, 2026"
        },
        {
            "id": "btc_110k_no",  # COMPLEMENT
            "event_type": "price_threshold",
            "underlying": "BTC",
            "operator": "<",
            "threshold": 110000,
            "deadline": "2026-06-30T23:59:59Z",
            "timezone": "UTC",
            "source": "Chainlink BTC/USD",
            "payout_condition": "YES iff BTC never reaches 110000 before deadline"
        },

        # ===== BTC THRESHOLD LADDER (SUBSET / SUPERSET) =====
        {
            "id": "btc_100k",
            "event_type": "price_threshold",
            "underlying": "BTC",
            "operator": ">=",
            "threshold": 100000,
            "deadline": "2026-06-30T23:59:59Z",
            "timezone": "UTC",
            "source": "Chainlink BTC/USD",
            "payout_condition": "YES iff BTC >= 100000 before deadline"
        },
        {
            "id": "btc_120k",  # SUBSET of btc_110k
            "event_type": "price_threshold",
            "underlying": "BTC",
            "operator": ">=",
            "threshold": 120000,
            "deadline": "2026-06-30T23:59:59Z",
            "timezone": "UTC",
            "source": "Chainlink BTC/USD",
            "payout_condition": "YES iff BTC >= 120000 before deadline"
        },

        # ===== TIME VARIATION EDGE CASE =====
        {
            "id": "btc_110k_short_window",  # NOT identical (shorter window)
            "event_type": "price_threshold",
            "underlying": "BTC",
            "operator": ">=",
            "threshold": 110000,
            "deadline": "2026-05-30T23:59:59Z",
            "timezone": "UTC",
            "source": "Chainlink BTC/USD",
            "payout_condition": "YES iff BTC >= 110000 before May 30"
        },

        # ===== ETH CLUSTER (UNRELATED vs BTC) =====
        {
            "id": "eth_6k_a",
            "event_type": "price_threshold",
            "underlying": "ETH",
            "operator": ">=",
            "threshold": 6000,
            "deadline": "2026-06-30T23:59:59Z",
            "timezone": "UTC",
            "source": "Chainlink ETH/USD",
            "payout_condition": "YES iff ETH >= 6000 before deadline"
        },
        {
            "id": "eth_6k_b",  # IDENTICAL
            "event_type": "price_threshold",
            "underlying": "ETH",
            "operator": ">=",
            "threshold": 6000,
            "deadline": "2026-07-01T00:00:00Z",
            "timezone": "UTC",
            "source": "Chainlink ETH/USD",
            "payout_condition": "YES iff ETH crosses 6000 before July 1"
        },

        # ===== OPERATOR EDGE CASE =====
        {
            "id": "btc_strict_gt",  # subtle difference from >=
            "event_type": "price_threshold",
            "underlying": "BTC",
            "operator": ">",
            "threshold": 110000,
            "deadline": "2026-06-30T23:59:59Z",
            "timezone": "UTC",
            "source": "Chainlink BTC/USD",
            "payout_condition": "YES iff BTC strictly exceeds 110000"
        },

        # ===== ELECTION CLUSTER =====
        {
            "id": "trump_win",
            "event_type": "election",
            "underlying": "Donald Trump",
            "operator": None,
            "threshold": None,
            "deadline": "2028-11-07T23:59:59Z",
            "timezone": "UTC",
            "source": "AP News",
            "payout_condition": "YES iff Donald Trump wins the 2028 US presidential election"
        },
        {
            "id": "trump_lose",  # COMPLEMENT
            "event_type": "election",
            "underlying": "Donald Trump",
            "operator": None,
            "threshold": None,
            "deadline": "2028-11-07T23:59:59Z",
            "timezone": "UTC",
            "source": "AP News",
            "payout_condition": "YES iff Donald Trump does not win the election"
        },

        # ===== CPI / MACRO =====
        {
            "id": "cpi_above_4",
            "event_type": "macro_event",
            "underlying": "US CPI YoY",
            "operator": ">=",
            "threshold": 4.0,
            "deadline": "2026-03-31T13:30:00Z",
            "timezone": "UTC",
            "source": "BLS",
            "payout_condition": "YES iff CPI YoY >= 4.0 for March release"
        },
        {
            "id": "cpi_above_3",  # SUPERSET of above_4
            "event_type": "macro_event",
            "underlying": "US CPI YoY",
            "operator": ">=",
            "threshold": 3.0,
            "deadline": "2026-03-31T13:30:00Z",
            "timezone": "UTC",
            "source": "BLS",
            "payout_condition": "YES iff CPI YoY >= 3.0"
        },

        # ===== SPORTS =====
        {
            "id": "lakers_win",
            "event_type": "sports",
            "underlying": "Lakers vs Warriors",
            "operator": None,
            "threshold": None,
            "deadline": "2026-05-01T03:00:00Z",
            "timezone": "UTC",
            "source": "NBA official",
            "payout_condition": "YES iff Lakers win the game"
        },
        {
            "id": "warriors_win",  # MUTUALLY EXCLUSIVE
            "event_type": "sports",
            "underlying": "Lakers vs Warriors",
            "operator": None,
            "threshold": None,
            "deadline": "2026-05-01T03:00:00Z",
            "timezone": "UTC",
            "source": "NBA official",
            "payout_condition": "YES iff Warriors win the game"
        },

        # ===== RANDOM UNRELATED =====
        {
            "id": "gold_3000",
            "event_type": "price_threshold",
            "underlying": "Gold",
            "operator": ">=",
            "threshold": 3000,
            "deadline": "2026-12-31T23:59:59Z",
            "timezone": "UTC",
            "source": "LBMA",
            "payout_condition": "YES iff gold >= 3000 before year end"
        }
    ]