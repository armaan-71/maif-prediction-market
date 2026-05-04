"""
Relationship Classifier for Prediction Markets

This module uses an LLM to determine if two markets are:
- IDENTICAL
- COMPLEMENT
- UNRELATED

Provider is configurable: set LLM_PROVIDER=openai or LLM_PROVIDER=groq in .env,
or pass provider= to MarketClassifier directly.

  openai  → uses OPENAI_API_KEY, defaults to gpt-4o-mini
  groq    → uses GROQ_API_KEY,   defaults to llama-3.3-70b-versatile
"""

import json
import logging
import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from models import UnifiedMarket

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Provider config ──────────────────────────────────────────────────────────

_PROVIDERS = {
    "openai": {
        "base_url": None,  # use the OpenAI SDK default
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
}

# ─── Structured output model ──────────────────────────────────────────────────

class RelationHypothesis(BaseModel):
    label: Literal["IDENTICAL", "COMPLEMENT", "UNRELATED"] = Field(
        ..., description="the logical relationship between the two markets"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    differences: List[str] = Field(default_factory=list, description="Key differences found")
    reason: str = Field(default="", description="The logic behind this classification")

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a semantic contract comparison engine for prediction markets.
Your job is to decide whether two prediction market contracts are IDENTICAL, COMPLEMENT, or UNRELATED.

CLASSIFICATIONS:
- IDENTICAL: Both resolve YES under exactly the same real-world conditions (same event, subject,
  and resolution threshold). Minor wording differences are fine; only the resolution *outcome* must
  match. Cross-exchange pairs always have different resolution infrastructure — ignore that.
- COMPLEMENT: YES on Market A means NO on Market B and vice versa, AND the two outcomes are
  EXHAUSTIVE — exactly one of the two MUST resolve YES (no third outcome is possible).
  Example: "Will Team A win?" vs "Will Team B win?" in a two-team playoff game with no draw.
  WARNING — NOT COMPLEMENT: "Fed cuts 25bps at June meeting" vs "Fed cuts >25bps at June meeting"
  is NOT complement because the Fed could also hold rates (both resolve NO). Use UNRELATED instead.
  WARNING — NOT COMPLEMENT: Any pair where both markets could resolve NO simultaneously.
- UNRELATED: Use for everything else — different events, different subjects, different scopes,
  subset/superset relationships, mutually exclusive but non-exhaustive pairs, or any other case
  that is not strictly IDENTICAL or COMPLEMENT.

CRITICAL — THRESHOLD vs. NARRATIVE TRAPS (most common error):
A threshold question like "Will X be ABOVE 0%?" resolves YES when X > 0 (positive outcome).
A narrative question like "Negative X in 2026?" resolves YES when X < 0 (negative outcome).
These are COMPLEMENT or UNRELATED, NOT IDENTICAL — the YES conditions point in opposite directions.
Before assigning IDENTICAL, explicitly state: "Market A resolves YES when [condition A]. Market B
resolves YES when [condition B]. These conditions are the same." If they differ in direction,
threshold, scope, or timeframe, use COMPLEMENT or UNRELATED.

CRITICAL — COMPLEMENT REQUIRES EXHAUSTIVENESS:
Before assigning COMPLEMENT, explicitly verify: "Is it IMPOSSIBLE for both markets to resolve NO
simultaneously?" If a third outcome exists (e.g., Fed holds rates, game is cancelled, tie is
possible), the pair is UNRELATED, not COMPLEMENT.

RULES:
1. Do NOT use any label other than IDENTICAL, COMPLEMENT, or UNRELATED.
2. If you are unsure, use UNRELATED.
3. Always populate the `reason` field with one sentence justifying your label.
4. Always include a `confidence` field: a float between 0.0 and 1.0 reflecting how certain you are.
5. Your `reason` must be consistent with your `label`. Do not say "different" or "opposite" for IDENTICAL.
6. For COMPLEMENT, your `reason` must state explicitly why the outcomes are exhaustive.
"""

# ─── Classifier ───────────────────────────────────────────────────────────────

class MarketClassifier:
    """
    LLM-based pair classifier.

    Provider selection (in priority order):
      1. `provider` constructor argument
      2. LLM_PROVIDER environment variable
      3. Defaults to "groq" for back-compat

    Model selection (in priority order):
      1. `model` constructor argument
      2. LLM_MODEL environment variable
      3. Provider default (gpt-4o-mini for openai, llama-3.3-70b-versatile for groq)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        resolved_provider = (
            provider
            or os.getenv("LLM_PROVIDER", "groq")
        ).lower()

        if resolved_provider not in _PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider '{resolved_provider}'. "
                f"Choose one of: {list(_PROVIDERS)}"
            )

        cfg = _PROVIDERS[resolved_provider]
        self.provider = resolved_provider
        self.model = model or os.getenv("LLM_MODEL") or cfg["default_model"]

        resolved_key = api_key or os.getenv(cfg["env_key"])
        if not resolved_key:
            raise ValueError(
                f"No API key found for provider '{resolved_provider}'. "
                f"Set {cfg['env_key']} in your .env or pass api_key= explicitly."
            )

        client_kwargs: dict = {"api_key": resolved_key}
        if cfg["base_url"]:
            client_kwargs["base_url"] = cfg["base_url"]

        self.client = OpenAI(**client_kwargs)
        logger.info(f"MarketClassifier using provider={self.provider}, model={self.model}")

    def classify_pair(
        self, market_a: UnifiedMarket, market_b: UnifiedMarket
    ) -> RelationHypothesis:
        """Compare two UnifiedMarket objects and return a RelationHypothesis."""
        info_a = self._get_semantic_context(market_a)
        info_b = self._get_semantic_context(market_b)

        user_prompt = (
            "Compare these two markets and return a JSON object following "
            "the RelationHypothesis schema.\n\n"
            f"MARKET A:\n{json.dumps(info_a, indent=2)}\n\n"
            f"MARKET B:\n{json.dumps(info_b, indent=2)}"
        )

        result = self._call_llm(user_prompt, SYSTEM_PROMPT + "\nReturn ONLY valid JSON.")

        # If the LLM omitted confidence (sentinel 0.0), retry once with an explicit reminder.
        if result is not None and result.confidence < 0.05 and result.label != "UNRELATED":
            logger.warning(
                f"LLM returned confidence={result.confidence} — retrying with explicit reminder"
            )
            reminder_system = (
                SYSTEM_PROMPT
                + "\nReturn ONLY valid JSON. You MUST include a `confidence` float in [0.0, 1.0]."
            )
            retry = self._call_llm(user_prompt, reminder_system)
            if retry is not None and retry.confidence >= 0.05:
                result = retry

        if result is None:
            return RelationHypothesis(
                label="UNRELATED",
                confidence=0.0,
                differences=["LLM call failed after retry"],
                reason="System failure during LLM call.",
            )
        return result

    def _call_llm(self, user_prompt: str, system_prompt: str) -> Optional["RelationHypothesis"]:
        """Single LLM call returning a parsed RelationHypothesis, or None on failure."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=30.0,
            )
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)
            # LLMs sometimes return "relation" or "classification" instead of "label"
            if "label" not in parsed:
                for alias in ("relation", "classification", "relationship"):
                    if alias in parsed:
                        parsed["label"] = parsed.pop(alias)
                        break
            return RelationHypothesis(**parsed)
        except Exception as e:
            logger.error(f"LLM classification failed ({self.provider}/{self.model}): {e}")
            return None

    def _get_semantic_context(self, market: UnifiedMarket) -> dict:
        """Extracts only the fields relevant for semantic comparison."""
        return {
            "question": market.question,
            "description": market.description,
            "outcomes": [o.label for o in market.outcomes],
            "close_at": market.close_at.isoformat() if market.close_at else "Unknown",
            "rules": market.resolution_rules,
            "event_title": market.event_title,
        }


if __name__ == "__main__":
    print("MarketClassifier module loaded.")
    print(f"Available providers: {list(_PROVIDERS)}")
    print(f"Active provider (from env): {os.getenv('LLM_PROVIDER', 'groq')}")
