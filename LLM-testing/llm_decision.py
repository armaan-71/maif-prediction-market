"""
Standalone LLM-based contract comparison engine (evaluation harness).

Uses a richer 7-label taxonomy than the pipeline classifier:
  IDENTICAL | COMPLEMENT | SUBSET | SUPERSET | MUTUALLY_EXCLUSIVE | UNRELATED | AMBIGUOUS

compare()           — single call
compare_consistent() — N-vote majority consensus (reduces label variance at 3× cost)

Model is configurable: defaults to gpt-4o-mini; override with LLM_MODEL env var.
"""

import json
import os
from collections import Counter
from typing import Literal

import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

import test_data

dotenv.load_dotenv()

data = test_data.get_test_data()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

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
The YES outcome of one contract is equivalent to the NO outcome of the other, and vice versa,
AND the outcomes are exhaustive (exactly one MUST resolve YES — no third outcome is possible).

SUBSET:
Whenever contract A resolves YES, contract B must also resolve YES, but not necessarily the reverse.

SUPERSET:
Whenever contract B resolves YES, contract A must also resolve YES, but not necessarily the reverse.

MUTUALLY_EXCLUSIVE:
Both contracts cannot resolve YES at the same time, but both could resolve NO (non-exhaustive).

UNRELATED:
The contracts do not have a meaningful deterministic semantic relationship.

AMBIGUOUS:
The contracts may be related, but the available information is insufficient to safely classify them.
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
9. Whether the outcome sets are exact inverses AND exhaustive
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

You must reason internally by:
1. extracting the core condition of contract 1
2. extracting the core condition of contract 2
3. comparing the critical fields
4. determining the safest supported relation label
5. listing any important differences or ambiguities

Confidence guidance:
- High confidence only when the evidence is direct and strong.
- Lower confidence when rules are incomplete, wording is vague, or equivalence depends on interpretation.
- If ambiguity is material, use AMBIGUOUS rather than a stronger label.

---

Few-shot examples:

Contract A: "Will the S&P 500 reach 6,500 at any point during Q1 2027?" | Source: NYSE intraday feed
Contract B: "Will the S&P 500 close at or above 6,500 on any trading day in Q1 2027?" | Source: NYSE official daily close
Label: AMBIGUOUS
Reason: Same index, threshold, and time window, but intraday touch vs. official daily close are materially different resolution conditions.

Contract A: "Will the Federal Reserve raise interest rates at the May 2027 FOMC meeting?" | Source: Federal Reserve official press release
Contract B: "Will the Federal Reserve hold or cut interest rates at the May 2027 FOMC meeting?" | Source: Federal Reserve official press release
Label: COMPLEMENT
Reason: The FOMC has exactly three outcomes — raise, hold, or cut — and contract B is explicitly the union of the two non-raise outcomes, making it the logical complement of A. YES on A is exactly NO on B. Same event, deadline, and source. The outcomes are exhaustive.

Contract A: "Will the Eurozone unemployment rate fall below 5.5% in the January 2027 release?" | Source: Eurostat
Contract B: "Will the Eurozone unemployment rate fall below 6.0% in the January 2027 release?" | Source: Eurostat
Label: SUBSET
Reason: Contract A has a stricter threshold. Whenever A resolves YES, B must also resolve YES — but not vice versa.

Contract A: "Will the Lakers win?" | Source: NBA official
Contract B: "Will the Warriors win?" | Source: NBA official
Label: MUTUALLY_EXCLUSIVE
Reason: Both teams cannot win simultaneously, but the game could theoretically be voided (both resolve NO). Not COMPLEMENT.
"""

client = OpenAI()


class RelationHypothesis(BaseModel):
    label: Literal[
        "IDENTICAL", "COMPLEMENT", "SUBSET", "SUPERSET",
        "MUTUALLY_EXCLUSIVE", "UNRELATED", "AMBIGUOUS"
    ] = Field(..., description="the relation between the 2 contracts")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Confidence in [0.0, 1.0]")
    differences: list[str] = Field(..., description="Notable differences between the two contracts")
    reason: str = Field(..., description="The reasoning behind the label")


SEMANTIC_FIELDS = [
    "title", "description", "rules", "outcomes", "resolution_time",
    "category", "entity_tags", "venue", "deadline",
    # test_data fields
    "event_type", "underlying", "operator", "threshold",
    "timezone", "source", "payout_condition",
]


def get_important_info(contract: dict) -> dict:
    return {k: contract[k] for k in SEMANTIC_FIELDS if k in contract}


def compare(contract1: dict, contract2: dict, model: str = DEFAULT_MODEL) -> RelationHypothesis:
    """Single LLM call — returns a RelationHypothesis."""
    info1 = get_important_info(contract1)
    info2 = get_important_info(contract2)
    user_message = (
        "Compare these two contracts:\n\n"
        f"CONTRACT 1:\n{json.dumps(info1, indent=2)}\n\n"
        f"CONTRACT 2:\n{json.dumps(info2, indent=2)}"
    )
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=RelationHypothesis,
        temperature=0.1,
    )
    return response.choices[0].message.parsed


def compare_consistent(
    contract1: dict, contract2: dict, model: str = DEFAULT_MODEL, n: int = 3
) -> RelationHypothesis:
    """N-vote majority consensus — reduces label variance at 3× cost."""
    results = [compare(contract1, contract2, model=model) for _ in range(n)]
    label_counts = Counter(r.label for r in results)
    majority_label, majority_count = label_counts.most_common(1)[0]
    agreement_ratio = majority_count / n
    best = max(
        (r for r in results if r.label == majority_label),
        key=lambda r: r.confidence,
    )
    best.confidence = round(best.confidence * agreement_ratio, 3)
    return best
