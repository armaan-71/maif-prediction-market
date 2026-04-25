"""
Quick test for classify_pair() — no Qdrant needed, just the LLM.
Run: python test_matcher.py
"""
import os
from dotenv import load_dotenv
from groq import Groq
from semantic_matcher import classify_pair

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

pairs = [
    (
        "Will Bitcoin exceed $100,000 before July 2026?",
        "BTC above 100k before July?",
        "kalshi", "polymarket",
        "IDENTICAL"
    ),
    (
        "Will the Lakers win tonight?",
        "Will the Warriors win tonight?",
        "kalshi", "polymarket",
        "MUTUALLY_EXCLUSIVE"
    ),
    (
        "Will Trump win the 2028 election?",
        "Will Trump lose the 2028 election?",
        "kalshi", "polymarket",
        "COMPLEMENT"
    ),
    (
        "Will it rain in Boston tomorrow?",
        "Will Bitcoin exceed $100,000?",
        "kalshi", "polymarket",
        "UNRELATED"
    ),
]

correct = 0
for text_a, text_b, ex_a, ex_b, expected in pairs:
    relation, confidence, reasoning = classify_pair(client, text_a, text_b, ex_a, ex_b)
    passed = relation == expected
    if passed:
        correct += 1
    print(f"{'✅' if passed else '❌'} Expected {expected} → Got {relation} ({confidence:.2f})")
    print(f"   Reasoning: {reasoning}\n")

print(f"Score: {correct}/{len(pairs)}")