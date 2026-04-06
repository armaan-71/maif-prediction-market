"""
Unified data model for prediction market contracts across exchanges.

Designed to normalize heterogeneous contract data from Kalshi, Polymarket,
Manifold, and Metaculus into a single schema suitable for semantic embedding
and cross-exchange arbitrage detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import json
import hashlib


# ─── Enums ────────────────────────────────────────────────────────────────────

class Exchange(str, Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"
    MANIFOLD = "manifold"
    METACULUS = "metaculus"


class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    UNKNOWN = "unknown"


class OutcomeType(str, Enum):
    BINARY = "binary"           # Yes/No
    MULTI = "multi"             # Multiple mutually-exclusive outcomes
    SCALAR = "scalar"           # Numeric range (Metaculus)


class ResolutionSource(str, Enum):
    CENTRALIZED_ORACLE = "centralized_oracle"    # Kalshi (CFTC-regulated staff)
    UMA_ORACLE = "uma_oracle"                    # Polymarket (decentralized)
    COMMUNITY = "community"                      # Manifold (creator + community)
    CROWD_FORECAST = "crowd_forecast"            # Metaculus (scoring, not market)
    UNKNOWN = "unknown"


# ─── Core Data Model ──────────────────────────────────────────────────────────

@dataclass
class Outcome:
    """A single tradable outcome within a market."""
    label: str                          # "Yes", "No", "Trump", "Harris", etc.
    price: Optional[float] = None       # 0.0-1.0 implied probability
    token_id: Optional[str] = None      # Polymarket CLOB token ID
    index: int = 0                      # Position in outcome array


@dataclass
class UnifiedMarket:
    """
    Canonical representation of a single prediction market contract.

    This is the unit that gets embedded. Every field that carries semantic
    signal about "what question is being asked" is kept as rich text.
    Pricing/volume fields are kept separate for the trading engine.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    exchange: Exchange
    native_id: str                      # Exchange-native identifier
    slug: Optional[str] = None          # Human-readable URL slug

    # ── Semantic Content (→ embedding input) ──────────────────────────────
    question: str = ""                  # The core question text
    description: str = ""               # Extended description / rules
    category: str = ""                  # Exchange-assigned category
    tags: list[str] = field(default_factory=list)

    # ── Event Grouping ────────────────────────────────────────────────────
    event_id: Optional[str] = None      # Parent event/series ID
    event_title: Optional[str] = None   # Parent event title
    group_id: Optional[str] = None      # Series ticker (Kalshi) or event slug (Poly)

    # ── Outcome Structure ─────────────────────────────────────────────────
    outcome_type: OutcomeType = OutcomeType.BINARY
    outcomes: list[Outcome] = field(default_factory=list)

    # ── Pricing Snapshot ──────────────────────────────────────────────────
    yes_price: Optional[float] = None   # Implied prob of primary outcome
    no_price: Optional[float] = None
    last_price: Optional[float] = None
    spread: Optional[float] = None      # Best ask - best bid
    volume_total: Optional[float] = None
    volume_24h: Optional[float] = None
    liquidity: Optional[float] = None
    open_interest: Optional[float] = None

    # ── Timing ────────────────────────────────────────────────────────────
    created_at: Optional[datetime] = None
    open_at: Optional[datetime] = None
    close_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # ── Resolution ────────────────────────────────────────────────────────
    status: MarketStatus = MarketStatus.UNKNOWN
    resolution_source: ResolutionSource = ResolutionSource.UNKNOWN
    resolution_rules: str = ""          # How this contract settles
    result: Optional[str] = None        # "yes"/"no"/outcome label if settled

    # ── Metadata ──────────────────────────────────────────────────────────
    url: Optional[str] = None
    fetched_at: Optional[datetime] = None

    # ── Derived ───────────────────────────────────────────────────────────

    @property
    def uid(self) -> str:
        """Deterministic unique ID: exchange + native_id."""
        return f"{self.exchange.value}:{self.native_id}"

    @property
    def embedding_text(self) -> str:
        """
        Concatenation of semantically-meaningful fields for embedding.
        This is the string you pass to your embedding model.
        Deduplicates fields that carry the same content.
        """
        parts = [self.question]
        if self.description:
            parts.append(self.description[:500])  # Truncate long rules
        if self.event_title and self.event_title != self.question:
            parts.append(f"Event: {self.event_title}")
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        # Only add resolution_rules if it provides info beyond description
        if self.resolution_rules and self.resolution_rules != self.description:
            parts.append(f"Resolution: {self.resolution_rules[:300]}")
        # Only add outcome labels if they carry semantic signal
        # (skip trivial "Yes, No" which adds nothing for embedding)
        outcome_labels = [o.label for o in self.outcomes]
        trivial = {"Yes", "No", "yes", "no"}
        if outcome_labels and not all(l in trivial for l in outcome_labels):
            parts.append(f"Outcomes: {', '.join(outcome_labels)}")
        return " | ".join(parts)

    @property
    def content_hash(self) -> str:
        """Hash of semantic content for deduplication."""
        return hashlib.sha256(self.embedding_text.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["uid"] = self.uid
        d["embedding_text"] = self.embedding_text
        d["content_hash"] = self.content_hash
        # Serialize enums and datetimes
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, datetime):
                d[k] = v.isoformat()
        if "outcomes" in d:
            for o in d["outcomes"]:
                for ok, ov in o.items():
                    if isinstance(ov, Enum):
                        o[ok] = ov.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)