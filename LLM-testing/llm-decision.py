"""
Given the input should look something like this:
{
  "contract_id": "kalshi_xyz",
  "venue": "kalshi",
  "title": "Will BTC exceed 110k by June 30, 2026?",
  "description": "...",
  "rules": "...",
  "outcomes": ["YES", "NO"],
  "resolution_time": "2026-06-30T23:59:59Z",
  "category": "crypto",
  "entity_tags": ["BTC"],
  "orderbook": {...},
  "fees": {...},
  "rewards": {...}
}

and the output should contain

{
  "label": "IDENTICAL",
  "confidence": 0.88,
  "differences": [],
  "critical_fields_checked": {
    "entity": true,
    "time_window": true,
    "threshold": true,
    "resolution_source": true
  },
  "reason": "Both resolve YES iff BTC trades at or above 110k before June 30, 2026 UTC."
}


We will also assume that the NLP portion of the pipeline has already determined a potential relation between the 2 contracts (idential/complement)
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
from collections import Counter
import json
import test_data
import dotenv

dotenv.load_dotenv()
data = test_data.get_test_data()

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
Both contracts cannot resolve YES at the same time.

UNRELATED:
The contracts do not have a meaningful deterministic semantic relationship.

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
Reason: Same index, threshold, and time window, but intraday touch vs. official daily close are materially different resolution conditions. An intraday wick to 6,500 resolves A YES but would not necessarily resolve B YES. Cannot classify as IDENTICAL.

Contract A: "Will the Federal Reserve raise interest rates at the May 2027 FOMC meeting?" | Source: Federal Reserve official press release
Contract B: "Will the Federal Reserve hold or cut interest rates at the May 2027 FOMC meeting?" | Source: Federal Reserve official press release
Label: COMPLEMENT
Reason: The FOMC has exactly three outcomes — raise, hold, or cut — and contract B is explicitly the union of the two non-raise outcomes, making it the logical complement of A. YES on A is exactly NO on B. Same event, deadline, and source.

Contract A: "Will Apple (AAPL) close above $250 on any trading day before December 31, 2026?" | Source: NASDAQ official closing price
Contract B: "Will Apple (AAPL) close above $250 on any trading day before December 31, 2026?" | Source: Bloomberg composite price
Label: AMBIGUOUS
Reason: Identical underlying, threshold, operator, and deadline, but NASDAQ official close and Bloomberg composite can diverge due to after-hours adjustments and data normalization. Cannot confirm IDENTICAL without verifying oracle agreement.

Contract A: "Will the Eurozone unemployment rate fall below 5.5% in the January 2027 release?" | Source: Eurostat
Contract B: "Will the Eurozone unemployment rate fall below 6.0% in the January 2027 release?" | Source: Eurostat
Label: SUBSET
Reason: Contract A has a stricter threshold (5.5% < 6.0%). Whenever A resolves YES (rate < 5.5%), B must also resolve YES (rate < 6.0%) — but B can resolve YES without A. A is the narrower condition, so A is a subset of B. Same source, event, and deadline.

Contract A: "Will the Nikkei 225 gain more than 5% in a single trading day before March 31, 2027?" | Source: Tokyo Stock Exchange
Contract B: "Will the Nikkei 225 gain more than 2% in a single trading day before March 31, 2027?" | Source: Tokyo Stock Exchange
Label: SUPERSET
Reason: Contract B has the stricter threshold (2% < 5%). Whenever B resolves YES (gain > 5%), A must also resolve YES (gain > 2%) — but A can resolve YES with a 3% gain while B would not. B is the narrower condition (subset), so A is the superset. Same index, source, and deadline.

Contract A: "Will Tesla (TSLA) stock close above $400 before June 30, 2027?" | Source: NASDAQ
Contract B: "Will Tesla (TSLA) stock close above $400 before March 31, 2027?" | Source: NASDAQ
Label: SUPERSET
Reason: Contract B has a shorter time window (ends March 31) contained within A's window (ends June 30). Whenever B resolves YES, A must also resolve YES since the same price event satisfies A's longer window — but A can resolve YES between April 1 and June 30 while B cannot. B is a subset of A, so A is the superset.
"""

client = OpenAI()


class RelationHypothesis(BaseModel):
    label: Literal["IDENTICAL", "COMPLEMENT", "SUBSET", "SUPERSET", "MUTUALLY_EXCLUSIVE", "UNRELATED", "AMBIGUOUS"] = Field(..., description="the relation between the 2 contracts")
    confidence: float = Field(..., ge = 0.0, le = 1.0, description="Value between 0.0 and 1.0 indicating your confidence of the label")
    differences: list[str] = Field(..., description="Any notable differences between the two contracts")
    reason: str = Field(..., description="The reasoning behind the label and your decision")



SEMANTIC_FIELDS = [
    # Real fields
    "title", "description", "rules", "outcomes", "resolution_time", "category", "entity_tags", "venue", "deadline"
    # Test data fields
    "event_type", "underlying", "operator", "threshold", "deadline", "timezone", "source", "payout_condition", "deadline"
]

def get_important_info(contract: dict):
    return {k: contract[k] for k in SEMANTIC_FIELDS if k in contract}


def compare(contract1, contract2, model: str = "gpt-5.4"):
    info1 = get_important_info(contract1)
    info2 = get_important_info(contract2)
    user_message = (
        f"Compare these two contracts:\n\n"
        f"CONTRACT 1:\n{json.dumps(info1, indent=2)}\n\n"
        f"CONTRACT 2:\n{json.dumps(info2, indent=2)}"
    )
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        text_format=RelationHypothesis,
        temperature=0.1,
    )
    return response.output_parsed


def compare_consistent(contract1, contract2, model: str = "gpt-5.4", n: int = 3):
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

