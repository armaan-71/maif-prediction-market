"""
Relationship Classifier for Prediction Markets

This module uses an LLM to determine if two markets are:
- IDENTICAL
- COMPLEMENT
- SUBSET / SUPERSET
- UNRELATED / AMBIGUOUS

It maps UnifiedMarket objects to a structured relationship hypothesis.
"""

import json
import logging
import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from models import UnifiedMarket

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)

# --- Structured Output Models ---

class RelationHypothesis(BaseModel):
    label: Literal["IDENTICAL", "COMPLEMENT", "SUBSET", "SUPERSET", "MUTUALLY_EXCLUSIVE", "UNRELATED", "AMBIGUOUS"] = Field(
        ..., description="the logical relationship between the two markets"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    differences: List[str] = Field(..., description="Key differences found")
    reason: str = Field(..., description="The logic behind this classification")

# --- Prompts ---

SYSTEM_PROMPT = """
You are a semantic contract comparison engine for prediction markets.
Your job is to compare two prediction market contracts and determine their logical relationship.

CLASSIFICATIONS:
- IDENTICAL: Both resolve YES under the exact same real-world conditions (same event, time, and threshold).
- COMPLEMENT: YES on Market A means NO on Market B (e.g., "Will it rain?" vs "Will it be dry?").
- SUBSET: If A is YES, B MUST be YES (A is narrower, e.g., "BTC > 100k" is a subset of "BTC > 90k").
- SUPERSET: If B is YES, A MUST be YES (A is broader).
- MUTUALLY_EXCLUSIVE: They cannot both be YES, but could both be NO.
- UNRELATED: No clear logical connection.
- AMBIGUOUS: They seem related but the fine print (dates, oracles) is too different to be sure.

CRITICAL CHECKLIST:
1. Event/Subject (e.g., Bitcoin, S&P 500)
2. Threshold & Operator (e.g., "> 100" vs ">= 100")
3. Deadline & Timezone (e.g., June 30 vs June 29)
4. Resolution Source (e.g., "Official Close" vs "Intraday Touch")

If you are unsure due to missing info, use AMBIGUOUS. Be conservative.
"""

# --- Classifier Class ---

class MarketClassifier:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.base_url = "https://api.groq.com/openai/v1"
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model

    def classify_pair(self, market_a: UnifiedMarket, market_b: UnifiedMarket) -> RelationHypothesis:
        """
        Compares two UnifiedMarket objects and returns a RelationHypothesis.
        """
        info_a = self._get_semantic_context(market_a)
        info_b = self._get_semantic_context(market_b)

        user_prompt = (
            f"Compare these two markets and return a JSON object following the RelationHypothesis schema.\n\n"
            f"MARKET A:\n{json.dumps(info_a, indent=2)}\n\n"
            f"MARKET B:\n{json.dumps(info_b, indent=2)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\nReturn ONLY valid JSON."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)
            return RelationHypothesis(**parsed)
        except Exception as e:
            logger.error(f"Groq/Llama Classification failed: {e}")
            return RelationHypothesis(
                label="AMBIGUOUS",
                confidence=0.0,
                differences=[f"Error: {str(e)}"],
                reason="System failure during LLM call."
            )

    def _get_semantic_context(self, market: UnifiedMarket) -> dict:
        """Extracts only the fields relevant for semantic comparison."""
        return {
            "question": market.question,
            "description": market.description,
            "outcomes": [o.label for o in market.outcomes],
            "close_at": market.close_at.isoformat() if market.close_at else "Unknown",
            "resolution_source": market.resolution_source.value,
            "rules": market.resolution_rules,
            "event_title": market.event_title
        }

# --- Quick Test ---
if __name__ == "__main__":
    # This is just a placeholder to show usage
    print("MarketClassifier module loaded.")
