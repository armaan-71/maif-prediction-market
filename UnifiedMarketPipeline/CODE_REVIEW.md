# Codebase Review — UnifiedMarketPipeline

**Date:** 2026-05-03  
**Scope:** Every Python file reviewed line-by-line.  
**Format:** CRITICAL > HIGH > MEDIUM > LOW. File + line numbers point to the current code.

---

## CRITICAL — Incorrect logic, crashes, or wrong financial results

---

### C1 · `backtest.py:343` — HEDGE_ARB closes on first settlement (`or` should be `and`)

```python
settled = (
    m_a.status == MarketStatus.SETTLED
    or m_b.status == MarketStatus.SETTLED   # ← BUG
)
```

For a hedge to pay $1 guaranteed, **both** legs must be settled. Closing after the first leg settles books a notional P&L before the second leg outcome is known. Change `or` → `and`. This hasn't caused visible wrong numbers yet only because none of our current markets have resolved — all trades are still "open" — but it will produce incorrect early-exit P&L the moment any market resolves.

---

### C2 · `pipeline.py:233` — `if args.summary or True:` always prints, `--summary` flag is dead

```python
if args.summary or True:  # Always print summary
```

The `or True` is a debug artifact. The flag has no effect; the summary always prints. Either remove the flag or remove the `or True`.

---

### C3 · `llm-decision.py:194` — `client.responses.parse()` does not exist in the OpenAI SDK

```python
response = client.responses.parse(
    model=model,
    input=[...],
    text_format=RelationHypothesis,
    ...
)
```

`client.responses` in the OpenAI v1 SDK is the Responses API (a different product, different signature). The `parse()` method for structured output lives at `client.beta.chat.completions.parse()`. This file crashes at runtime. The entire LLM-testing module is currently non-functional.

---

### C4 · `llm-decision.py:186` / `test-llm-decision.py:52` — Non-existent model names

Both files reference `gpt-5.4` and `gpt-5.4-mini`. These are not public OpenAI model identifiers. Every call returns HTTP 404 / `model_not_found`.

---

### C5 · `test-llm-decision.py:1` — Hyphenated module name cannot be imported reliably

```python
llm_decision = __import__('llm-decision')
```

Python module names with hyphens cannot be reliably imported this way on all platforms. The file needs to be renamed to `llm_decision.py` and imported as `import llm_decision`.

---

## HIGH — Bugs that produce wrong results or silently degrade quality

---

### H1 · `arbitrage_calculator.py:117–119` — Fee model is wrong for Kalshi flat-rate contracts

```python
raw_cost = p1 + p2
fee_cost = raw_cost * self.default_fee_rate
```

Kalshi charges approximately 7¢ per contract regardless of contract price. For a YES contract at $0.04, a 7¢ flat fee is a 175% surcharge — far higher than the 1–2% `fee_rate` the backtest uses. Polymarket charges 2% of **profit**, not of cost. The current model underestimates Kalshi fees on low-probability markets and overestimates Polymarket fees. Both of our live Fed June arb positions (entry costs $0.9865 and $0.9965) would be wiped at realistic Kalshi fees.

The fix requires per-exchange fee functions:
```python
def kalshi_fee(cost_per_contract: float, n_contracts: int) -> float:
    return 0.07 * n_contracts  # flat 7¢/contract

def polymarket_fee(payout: float, cost: float) -> float:
    return max(0.0, payout - cost) * 0.02  # 2% of net profit
```

---

### H2 · `scout.py:181–190` — Dedup gap: a Polymarket market can be paired with N Kalshi siblings

```python
# Block the same counterpart market from being paired with multiple source markets
if uid_b in uids and uid_a not in uids:
    return True
```

This guards `uid_b` (the candidate) but not `uid_a` (the source). When Polymarket is run as the source, a Polymarket market already paired with Kalshi market K_A can still be paired with Kalshi market K_C — because the check only fires when `uid_b` is already in an existing pair. Add the symmetric check:

```python
if uid_a in uids and uid_b not in uids:
    return True
```

---

### H3 · `adapters.py:23` — Module-level `logging.basicConfig` clobbers all later configs

```python
# in adapters.py (top-level, runs at import time):
logging.basicConfig(filename='app.log', level=logging.DEBUG)
```

Python's `basicConfig` is a no-op if any handlers are already configured. Since `adapters.py` is imported by `pipeline.py`, which is imported by `scout_pipeline.py`, this file-based handler is the first to register. The `logging.basicConfig(level=INFO, format=...)` calls in `scout.py:24`, `scout_pipeline.py:39`, and `backtest.py:420` are all silently ignored. Consequence: console output format in scout and backtest is uncontrolled, and all log levels are forced to DEBUG. Move this to a dedicated `setup_logging()` function called only in `if __name__ == "__main__"` blocks.

---

### H4 · `PolymarketAdapter.fetch_markets` — No rate limiting between pagination pages

```python
while True:
    resp = await self.client.get(f"{self.BASE}/markets", params=params)
    ...
    offset += page_size
```

There is no `await asyncio.sleep(...)` between pages. With `limit=5000` and page_size=100, this fires 50 API calls as fast as the async event loop allows. Polymarket's public endpoint will 429 under this load. The Kalshi series path has an `await asyncio.sleep(0.3)` inter-series pause; the Polymarket path needs an equivalent (0.1–0.2s per page is sufficient).

---

### H5 · `PolymarketAdapter._normalize:422` — `resolution_rules` contains the wrong field

```python
resolution_rules=m.get("resolutionSource", ""),
```

`resolutionSource` is a URL or name of the oracle (e.g., `"uma"`), not a description of resolution rules. The actual rules text lives in `m.get("description", "")` — but that's already mapped to `description`. The correct field for resolution criteria is likely `m.get("resolutionRules", "")` or left empty. As-is, the LLM classifier sees a URL string as "resolution rules" for every Polymarket market, degrading classification accuracy.

Also on the same function: `result=m.get("resolvedBy")` returns the oracle contract address, not "yes" / "no". The backtest's `_try_close` checks `leg_market.result` against `("yes", "y", "true", "1")`, which will never match an Ethereum address. DIRECTIONAL_GAP trades can never close via resolution for Polymarket markets.

---

### H6 · `ManifoldAdapter._normalize:509` — Zero probability treated as None

```python
no_price=(1.0 - prob) if prob else None,
```

`prob = 0.0` is falsy. A market with 0% probability (rare but valid) gets `no_price=None` instead of `1.0`. Change to `if prob is not None else None`.

---

### H7 · `ManifoldAdapter.fetch_markets` and `MetaculusAdapter.fetch_markets` — No pagination

Both adapters fetch exactly one page and return. Manifold supports cursor-based pagination via `before=<id>` and Metaculus via `offset=<int>`. With `limit=5000`, only the first 1000 (Manifold cap) or 100 (Metaculus cap) markets are ever retrieved.

---

## MEDIUM — Quality issues, misleading data, missing safeguards

---

### M1 · `llm_classifier.py:51` — `confidence` default of 0.9 silently stamps uncertain calls

```python
confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")
```

If the LLM omits `confidence`, every pair gets 0.9 stamped in. This passes the `min_confidence=0.7` gate. A sentinel value of `0.0` (or making the field required with no default) would surface which calls are missing confidence. In the current `verified_pairs.test.json`, all 5 BTC pairs show exactly `"confidence": 0.9` — these are all LLM-omitted values.

---

### M2 · `BacktestResult.roi:102` — Denominator includes open (unrealized) trades

```python
@property
def total_invested(self) -> float:
    return sum(t.entry_cost * t.position_size for t in self.trades)  # all trades

@property
def roi(self) -> float:
    return self.total_pnl / self.total_invested if self.total_invested else 0.0
```

`total_pnl` counts only closed trades; `total_invested` counts all trades including open ones. The ROI figure in the summary mixes realized and deployed capital, making it meaningless when most trades are still open (as in our current backtest). Add a `realized_roi` property that only counts closed trades.

---

### M3 · `backtest.py:361` — `result` field is never populated for historical snapshots

In `materialize_snapshots`, non-final snapshots have `snap.result = None`. The final snapshot inherits from the source market, which is a minimal stub from `markets_from_pairs` that sets no `result`. So `leg_market.result` is always `None`, and DIRECTIONAL_GAP trades can only close via convergence, never via resolution.

---

### M4 · `OddpoolKalshiFetcher.fetch:308` — Cent-to-dollar conversion threshold is fragile

```python
if yes_price > 1.5:
    yes_price /= 100.0  # Kalshi cents
```

A very-high-probability Kalshi market might return `yes_price = 0.99` (dollar) or `99.0` (cents). The `> 1.5` threshold handles this correctly for most cases. But a market at exactly $1.51 (impossible in practice since Kalshi caps at $0.99) would be incorrectly divided. The threshold is empirically fine for Kalshi's price range (1¢–99¢) but should have a comment explaining the assumption.

---

### M5 · `scout_pipeline.py:213` — `--accept-labels` help text lists 5 labels that the classifier doesn't emit

```
help="Comma-separated labels to accept (default: IDENTICAL,COMPLEMENT,SUBSET,SUPERSET,MUTUALLY_EXCLUSIVE)"
```

`llm_classifier.py`'s `RelationHypothesis` only accepts `IDENTICAL | COMPLEMENT | UNRELATED`. SUBSET, SUPERSET, MUTUALLY_EXCLUSIVE are not in the schema. The help text should read `"(default: IDENTICAL,COMPLEMENT)"`.

---

### M6 · `scout_pipeline.py:216` — `--llm-call-delay` default (2.0s) contradicts Scout class default (0.0s)

The CLI default is 2s, but if Scout is instantiated directly (e.g., in tests or scripts), the default is 0s. This inconsistency is likely to cause accidental throttling or unexpected rate-limit errors depending on which entry point is used.

---

### M7 · `historical_data.py:380` — Settled-market cache optimization never fires

```python
already_resolved = market.status == MarketStatus.SETTLED and cached
```

Markets loaded from `verified_pairs.json` via `markets_from_pairs` always have `status = MarketStatus.UNKNOWN` (default). The skip-if-settled optimization never triggers, so every `get_history` call hits the API even for resolved markets. Change `markets_from_pairs` to preserve status when available, or move this check to after the status is explicitly known.

---

### M8 · `vector_store.py:94` — Wrong UUID namespace (`NAMESPACE_DNS` used for market IDs)

```python
market_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
```

`NAMESPACE_DNS` is semantically for DNS hostnames. Market UIDs are application-specific identifiers. Use `uuid.NAMESPACE_URL` or `uuid.NAMESPACE_OID`. This doesn't cause incorrect behavior but is semantically wrong and could cause UUID collisions if the namespace semantics ever matter.

---

### M9 · `COMPLEMENT` HEDGE_ARB is not truly risk-free for non-exhaustive event spaces

In `arbitrage_calculator.py`, a COMPLEMENT hedge buys YES on both sides. This pays $1 iff exactly one of the two events occurs. But:

- Pair 8: "Fed cut 25bps at June" vs "Fed cut >25bps at June" — if the Fed makes NO move in June, both resolve NO → $0 payout, full loss.
- Pair 10: "No cuts in 2026" vs "Cut >25bps at December" — if the Fed cuts 25bps at December, "no cuts" = NO and "cut >25bps" = NO → both lose.

These are MUTUALLY_EXCLUSIVE pairs, not COMPLEMENT pairs. The LLM mislabeled them as COMPLEMENT. The calculator correctly identifies a HEDGE_ARB because the math works out if the assumption holds, but the assumption doesn't hold here. The classifier should distinguish COMPLEMENT (strict logical inverses) from MUTUALLY_EXCLUSIVE (can't both be YES, but both can be NO).

---

### M10 · `backtest.py:292–295` — `pair_id` in `_make_hedge_trade` comes from `opp`, not from `pair`

Looking at `_make_hedge_trade`, the returned Trade's `pair_id` comes from `pair["pair_id"]`. That's correct. But in `trades_output.json`, the `pair_id` in the open trades (`pair_20260502...`) doesn't match any `pair_id` in `verified_pairs.test.json` (`pair_20260503...`). This is because `trades_output.json` was generated from an older `verified_pairs.json` (pre-cleanup). Not a code bug, but worth noting that `pair_id` continuity is not guaranteed across scout runs.

---

## LOW — Code hygiene, minor improvements

---

### L1 · `adapters.py:428` — `import json` inside a `@staticmethod`

```python
@staticmethod
def _parse_json_array(val) -> list:
    import json
```

`json` is already in stdlib and very cheap to import. Move it to the module-level imports. There's no benefit to the local import.

---

### L2 · `historical_data.py:497` — `_parse_date` uses `replace(tzinfo=...)` instead of `astimezone`

```python
def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
```

If the input string contains timezone info (e.g., `"2025-01-01T00:00:00+05:30"`), `replace(tzinfo=timezone.utc)` overwrites it without converting, producing a wrong timestamp. Use:

```python
dt = datetime.fromisoformat(s)
return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
```

Same issue exists in `backtest.py:416`.

---

### L3 · `models.py:144` — Variable `l` looks like `1` in the outcome label comprehension

```python
if outcome_labels and not all(l in trivial for l in outcome_labels):
```

Rename `l` to `label` for readability.

---

### L4 · `pipeline.py:88–90` — Kalshi `series_tickers` guard is duplicated

```python
if series_tickers and exchange == Exchange.KALSHI:
    kwargs["series_tickers"] = series_tickers
```

This guard in `fetch_exchange` is duplicated by `run_pipeline:89` which already only passes `series_tickers=kalshi_series if exchange == Exchange.KALSHI else None`. One of the two guards is redundant.

---

### L5 · `scout.py:194` — `pair_id` timestamp collision possible within the same second

```python
"pair_id": f"pair_{datetime.now().strftime('%Y%m%d%H%M%S')}_{m_a.native_id}",
```

If two different source markets (both A1 and A2 from exchange X) produce matches within the same second, the pair_id is unique because it's scoped to `m_a.native_id`. However, if the SAME market A matches two different candidates B1 and B2 within one second, there would be two pairs with the same `pair_id`. Use `%Y%m%d%H%M%S%f` (microseconds) or a UUID.

---

### L6 · `.gitignore` missing common items

```
# Current .gitignore
qdrant_data/
data/historical/
```

Missing:
- `qdrant_test/` (test DB)
- `verified_pairs.test.json` (test output)
- `trades_output.json` (ephemeral backtest output)
- `*.log` (app.log files)
- `__pycache__/`
- `*.pyc`
- `.env`
- `bench_results/` (run_bench_arb output)

---

### L7 · `historical_data.py:174` — `latest_timestamp` loads the whole file every call

`HistoricalDataCache.latest_timestamp` calls `self.load()`, which reads the whole file. `cache.append()` calls `latest_timestamp()`. For large JSONL files, this is O(n) per append. Cache the last timestamp in memory or use `tail` logic (seek to end of file).

---

### L8 · `requirements.txt` is missing packages used in the codebase

The file lists 5 packages but doesn't include:
- `matplotlib` (optional, used in `run_bench_arb.py`)
- `groq` is not needed (uses OpenAI compat), but `oddpool` SDK (if used) is not listed either

---

### L9 · `LLM-testing/` is disconnected from `UnifiedMarketPipeline`

The richer 7-label taxonomy in `llm-decision.py` (IDENTICAL, COMPLEMENT, SUBSET, SUPERSET, MUTUALLY_EXCLUSIVE, UNRELATED, AMBIGUOUS) and the `compare_consistent` majority-vote approach are both more robust than the 3-label schema in `llm_classifier.py`. These improvements were never ported. Specifically:

- **AMBIGUOUS** label would prevent borderline pairs from being accepted as IDENTICAL/COMPLEMENT
- **SUBSET/SUPERSET** would correctly classify Kalshi threshold siblings vs. Polymarket binary questions (instead of letting them through as IDENTICAL)
- **`compare_consistent`** (n=3 majority vote) reduces label variance but costs 3× LLM calls

---

### L10 · `run_bench_arb.py:25–27` — Hardcoded sibling-repo path assumption

```python
_BENCH_SRC = _REPO_ROOT.parent / "PredictionMarketBench" / "src"
```

This fails if `PredictionMarketBench` is not a sibling directory. Should read from an environment variable or argparse `--bench-src` flag.

---

## Summary Table

| ID | File | Severity | Issue |
|----|------|----------|-------|
| C1 | `backtest.py:343` | CRITICAL | Hedge closes on first settlement, not both |
| C2 | `pipeline.py:233` | CRITICAL | `or True` kills `--summary` flag |
| C3 | `llm-decision.py:194` | CRITICAL | `client.responses.parse()` doesn't exist |
| C4 | `llm-decision.py:186` | CRITICAL | Model names `gpt-5.4` / `gpt-5.4-mini` don't exist |
| C5 | `test-llm-decision.py:1` | CRITICAL | Hyphenated filename can't be `__import__`ed |
| H1 | `arbitrage_calculator.py:117` | HIGH | Fee model wrong for Kalshi flat-rate contracts |
| H2 | `scout.py:181` | HIGH | Dedup gap: source uid not blocked when already paired |
| H3 | `adapters.py:23` | HIGH | Module-level `basicConfig` clobbers all other log config |
| H4 | `adapters.py:356` | HIGH | Polymarket pagination has no rate limiting |
| H5 | `adapters.py:422` | HIGH | `resolution_rules` maps to wrong field; `result` maps to oracle address |
| H6 | `adapters.py:509` | HIGH | `prob=0.0` treated as None (falsy bug) |
| H7 | `adapters.py:487/561` | HIGH | Manifold and Metaculus adapters have no pagination |
| M1 | `llm_classifier.py:51` | MEDIUM | `confidence` default 0.9 silently stamps uncertain calls |
| M2 | `backtest.py:102` | MEDIUM | ROI denominator includes open trades |
| M3 | `backtest.py:361` | MEDIUM | `result` is never populated; gap trades can't close via resolution |
| M4 | `historical_data.py:308` | MEDIUM | Cent-to-dollar threshold is fragile but empirically safe |
| M5 | `scout_pipeline.py:213` | MEDIUM | Wrong label list in `--accept-labels` help text |
| M6 | `scout_pipeline.py:216` | MEDIUM | CLI delay default (2.0s) inconsistent with Scout class default (0.0s) |
| M7 | `historical_data.py:380` | MEDIUM | Settled-market cache skip never fires |
| M8 | `vector_store.py:94` | MEDIUM | Wrong UUID namespace |
| M9 | `arbitrage_calculator.py` | MEDIUM | COMPLEMENT hedge not risk-free for non-exhaustive event spaces |
| M10 | `backtest.py` | MEDIUM | `pair_id` in trades doesn't match current pairs file after re-scout |
| L1 | `adapters.py:428` | LOW | `import json` inside staticmethod |
| L2 | `historical_data.py:497` | LOW | `replace(tzinfo=)` instead of conditional replace |
| L3 | `models.py:144` | LOW | Variable `l` looks like `1` |
| L4 | `pipeline.py:88` | LOW | Kalshi series guard is duplicated |
| L5 | `scout.py:194` | LOW | `pair_id` collision possible within same second |
| L6 | `.gitignore` | LOW | Missing qdrant_test/, *.log, *.pyc, etc. |
| L7 | `historical_data.py:174` | LOW | `latest_timestamp` re-reads file on every append |
| L8 | `requirements.txt` | LOW | Missing optional dependencies |
| L9 | `LLM-testing/` | LOW | Richer taxonomy never ported to pipeline classifier |
| L10 | `run_bench_arb.py:25` | LOW | Hardcoded sibling-repo path |

---

## Top 5 Recommended Fixes Before Deadline

Given the project goal (find and demonstrate real cross-exchange arb), these are the highest-leverage fixes:

1. **H1 (Fee model)**: The current backtest reports 7 "profitable" arbs but the two most plausible ones (Fed June pairs) are wiped by real Kalshi fees. Implementing per-contract Kalshi fees would give an accurate picture of what actually survives.

2. **C1 (Hedge close logic)**: Fix `or` → `and` in `_try_close`. No immediate impact since nothing has resolved, but it's a correctness bug that will produce wrong P&L numbers the moment any market settles.

3. **M9 (COMPLEMENT hedge is conditional)**: The 3 COMPLEMENT pairs (pairs 8, 9, 10) are mis-labeled as "HEDGE_ARB" (risk-free). They are conditional bets that pay only if one of the two outcomes occurs. This needs a label in the trade record and a different P&L accounting formula.

4. **H2 (Dedup gap)**: Without this fix, a second scout run could produce duplicate pairs if Polymarket is run as source after Kalshi.

5. **L6 (.gitignore)**: `verified_pairs.test.json` and `qdrant_test/` are not gitignored, meaning test state gets committed alongside production state.
