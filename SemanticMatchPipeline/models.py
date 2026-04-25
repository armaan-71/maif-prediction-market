from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MarketSummary(BaseModel):
    uid: str
    question: str
    exchange: str
    event_id: Optional[str] = None
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    close_at: Optional[str] = None
    resolution_rules: Optional[str] = None
    description: Optional[str] = None
    embedding_text: str


class MarketCluster(BaseModel):
    cluster_id: int
    markets: list[MarketSummary]


class RelationType(str, Enum):
    IDENTICAL = "IDENTICAL"
    COMPLEMENT = "COMPLEMENT"
    SUBSET = "SUBSET"
    SUPERSET = "SUPERSET"
    MUTUALLY_EXCLUSIVE = "MUTUALLY_EXCLUSIVE"
    UNRELATED = "UNRELATED"
    AMBIGUOUS = "AMBIGUOUS"


class ClassifiedPair(BaseModel):
    market_a: MarketSummary
    market_b: MarketSummary
    relation: RelationType
    confidence: float
    differences: list[str]
    reason: str


class ArbitrageOpportunity(BaseModel):
    pair: ClassifiedPair
    arb_edge: float
    strategy: str


class PipelineOutput(BaseModel):
    opportunities: list[ArbitrageOpportunity]
    unrelated_pairs: int
    ambiguous_pairs: int
    total_clusters: int
    total_pairs_classified: int
    generated_at: str
