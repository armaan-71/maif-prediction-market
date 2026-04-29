"""
LLM-based pairwise relation classification using Groq (llama-3.1-8b-instant).
"""
from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

from .models import ClassifiedPair, MarketCluster, MarketSummary, RelationType

GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
You are a semantic contract comparison engine for prediction markets.

Your job is to compare two prediction market contracts and produce a strict, structured relation hypothesis.
You do not make trading decisions.
You do not guess missing facts.
You do not smooth over ambiguities.
You do not assume two contracts are equivalent unless the important fields truly align.

Your goal is to determine the semantic relationship between the two contracts using only the information provided.

You must classify the relationship as exactly one of:

- IDENTICAL
- COMPLEMENT
- SUBSET
- SUPERSET
- MUTUALLY_EXCLUSIVE
- UNRELATED
- AMBIGUOUS

Definitions:

IDENTICAL:
Both contracts resolve to the same outcome under the same real-world conditions.
Use only if the underlying event, threshold/operator logic, effective time window, and resolution conditions are meaningfully equivalent.

COMPLEMENT:
The YES outcome of one contract is equivalent to the NO outcome of the other, and vice versa.

SUBSET:
Whenever contract A resolves YES, contract B must also resolve YES, but not necessarily the reverse.

SUPERSET:
Whenever contract B resolves YES, contract A must also resolve YES, but not necessarily the reverse.

MUTUALLY_EXCLUSIVE:
Both contracts cannot resolve YES at the same time, but both CAN resolve NO.
Use this when they cover different, non-overlapping bands or outcomes of the same underlying event.
Example: "sentenced to <5 years" and "sentenced to 5-10 years" are MUTUALLY_EXCLUSIVE — only one
band can be true. This is NOT UNRELATED; it is a meaningful logical constraint.

UNRELATED:
The contracts have NO deterministic logical constraint between them at all.
Use only when the two contracts concern genuinely different subjects or events with no shared resolution logic.
Do NOT use UNRELATED when the contracts clearly refer to the same event but cover different outcome bands —
that is MUTUALLY_EXCLUSIVE.

AMBIGUOUS:
The contracts may be related, but the available information is insufficient to safely classify them as one of the above.
When uncertain, prefer AMBIGUOUS.

Important comparison dimensions:
1. Underlying subject/entity
2. Event type
3. Threshold values and comparison operators
4. Time window and deadline
5. Timezone and effective cutoff
6. Resolution source / oracle / settlement rule
7. Whether wording means "by", "before", "on", "at close", "touches", "closes above", "wins", "leads", etc.
8. Whether one contract is broader or narrower than the other
9. Whether the outcome sets are exact inverses
10. Whether any missing or unclear rule prevents safe classification

Rules:
- Never assume missing details.
- If a key field is unknown or unclear, reflect that in the output.
- Small wording differences may be semantically critical.
- "By June 30" is not automatically identical to "on June 30".
- "Touches 100" is not automatically identical to "closes above 100".
- Different resolution sources can make contracts non-identical.
- If two contracts appear similar but not provably equivalent, output AMBIGUOUS.
- Prefer false negatives over false positives.
- Be conservative.

You must respond with valid JSON in this exact format:
{
  "relation": "<one of the 7 labels>",
  "confidence": <float 0.0-1.0>,
  "differences": ["<difference 1>", "..."],
  "reason": "<explanation>"
}
"""


def _format_contract(m: MarketSummary, label: str) -> str:
    parts = [
        f"CONTRACT {label}:",
        f"  Exchange: {m.exchange}",
        f"  Question: {m.question}",
    ]
    if m.description:
        parts.append(f"  Description: {m.description}")
    if m.resolution_rules:
        parts.append(f"  Resolution rules: {m.resolution_rules}")
    if m.close_at:
        parts.append(f"  Close at: {m.close_at}")
    if m.yes_price is not None:
        parts.append(f"  Yes price: {m.yes_price}")
    return "\n".join(parts)


def _should_skip_pair(
    a: MarketSummary,
    b: MarketSummary,
    cross_exchange_only: bool = True,
) -> bool:
    # Always skip same-exchange series contracts (different strikes of the same event)
    if a.exchange == b.exchange and a.event_id is not None and a.event_id == b.event_id:
        return True
    # By default only classify cross-exchange pairs — those are the arb targets
    if cross_exchange_only and a.exchange == b.exchange:
        return True
    return False


def classify_pair(
    market_a: MarketSummary,
    market_b: MarketSummary,
    client: Groq,
    retries: int = 3,
    backoff: float = 2.0,
) -> ClassifiedPair:
    user_message = (
        f"{_format_contract(market_a, 'A')}\n\n"
        f"{_format_contract(market_b, 'B')}\n\n"
        "Compare these two contracts and return JSON."
    )

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=30.0,
            )
            raw = json.loads(response.choices[0].message.content)
            break
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    else:
        print(f"[classifier] WARN: all {retries} attempts failed for pair "
              f"({market_a.uid}, {market_b.uid}): {last_err}")
        return ClassifiedPair(
            market_a=market_a,
            market_b=market_b,
            relation=RelationType.AMBIGUOUS,
            confidence=0.0,
            differences=[],
            reason=f"Classification failed after {retries} retries: {last_err}",
        )

    relation_str = raw.get("relation", "AMBIGUOUS").upper()
    try:
        relation = RelationType(relation_str)
    except ValueError:
        relation = RelationType.AMBIGUOUS

    return ClassifiedPair(
        market_a=market_a,
        market_b=market_b,
        relation=relation,
        confidence=float(raw.get("confidence", 0.5)),
        differences=raw.get("differences", []),
        reason=raw.get("reason", ""),
    )


def classify_cluster(
    cluster: MarketCluster,
    client: Groq,
    cross_exchange_only: bool = True,
    max_pairs: int = 0,
) -> list[ClassifiedPair]:
    """Classify all eligible pairs in a cluster. max_pairs=0 means no cap."""
    candidates = [
        (a, b)
        for i, a in enumerate(cluster.markets)
        for b in cluster.markets[i + 1:]
        if not _should_skip_pair(a, b, cross_exchange_only=cross_exchange_only)
    ]
    if max_pairs and len(candidates) > max_pairs:
        print(f"[classifier] Cluster {cluster.cluster_id}: capping {len(candidates)} → {max_pairs} pairs")
        candidates = candidates[:max_pairs]

    results: list[ClassifiedPair] = []
    for a, b in candidates:
        pair = classify_pair(a, b, client)
        results.append(pair)
    return results


def has_cross_exchange_pairs(cluster: MarketCluster) -> bool:
    exchanges = {m.exchange for m in cluster.markets}
    return len(exchanges) > 1


def make_groq_client() -> Groq:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key at https://console.groq.com"
        )
    return Groq(api_key=api_key)
