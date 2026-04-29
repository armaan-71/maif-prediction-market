# maif-prediction-market

A multi-agent NLP pipeline that semantically maps prediction market contracts across exchanges and identifies deterministic arbitrage pairs. Instead of taking directional bets, it acts as a liquidity provider by finding contracts that ask the same underlying question at different prices.

## How it works

```
UnifiedMarketPipeline          SemanticMatchPipeline
──────────────────────         ──────────────────────────────────────
Kalshi ─┐                      grouper.py       classifier.py
Poly   ─┼─► adapters.py ──► markets.json ──► Qdrant KNN ──► Groq LLM
Manifold┤                      clusters             │
Metaculus┘                                     arb_pairs.json
                                                     │
                                               poller.py (live prices)
                                                     │
                                               opportunities printed
```

1. **Ingest** — fetch all open markets from four exchanges, normalize into a unified schema, upsert embeddings into a local Qdrant vector DB
2. **Cluster** — embed each market's question text and run KNN search to group semantically similar markets across exchanges
3. **Classify** — send cross-exchange pairs to an LLM (Groq/Llama) which labels the semantic relationship between them
4. **Detect** — score classified pairs for price mismatch given their relationship; write confirmed candidates to `arb_pairs.json`
5. **Poll** — continuously fetch live quotes for confirmed pairs and alert when edge exceeds threshold

## Supported exchanges

| Exchange | Type | Auth required |
|---|---|---|
| Kalshi | Real-money, binary | No (read) |
| Polymarket | Real-money, binary | No (read) |
| Manifold | Play-money | No |
| Metaculus | Forecasting (no trading) | No |

All market categories are included — no category filter is applied. Kalshi MVE parlays (multi-leg combos) are excluded since they have no standalone semantic question.

## Detected relation types

| Relation | Arb condition |
|---|---|
| `IDENTICAL` | Same question priced differently across exchanges |
| `COMPLEMENT` | YES_A + YES_B < 1.0 (buy both for guaranteed payout) |
| `MUTUALLY_EXCLUSIVE` | YES_A + YES_B > 1.0 (sell both) |
| `SUBSET` | P(A) > P(B) when A ⊂ B (impossible by probability law) |
| `SUPERSET` | P(A) < P(B) when A ⊃ B (impossible by probability law) |

## Setup

**1. Install dependencies**

```bash
pip install httpx "qdrant-client[fastembed]" python-dotenv pydantic groq
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Then edit `.env` and fill in your values:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | LLM classifier. Free key at [console.groq.com](https://console.groq.com) |

All exchange APIs used for reading (Kalshi, Polymarket, Manifold, Metaculus) are public and require no authentication.

> **Never commit `.env`** — it is listed in `.gitignore`. Use `.env.example` as the source of truth for required variables.

## Running

### Step 1 — Ingest markets and build the vector DB

Run from the `UnifiedMarketPipeline/` directory:

```bash
cd UnifiedMarketPipeline
python pipeline.py --vector
```

This fetches up to 200 open markets per exchange, writes `markets.json`, and upserts embeddings into `qdrant_data/`.

Options:

```
--exchanges kalshi polymarket manifold metaculus   # subset of exchanges
--limit 500                                        # markets per exchange
--vector-batch-size 64                             # embedding batch size
--vector-parallel 0                                # CPU workers (0 = all cores)
--format json|jsonl|embedding|none                 # output format
```

### Step 2 — Run the semantic match pipeline

Run from the project root:

```bash
python -m SemanticMatchPipeline.pipeline
```

This clusters markets by vector similarity, classifies cross-exchange pairs via LLM, and writes two output files:

- `SemanticMatchPipeline/arb_pairs.json` — confirmed semantic pairs (no prices; used by the poller)
- `SemanticMatchPipeline/opportunities.json` — snapshot opportunities scored against prices at run time

Options:

```
--markets UnifiedMarketPipeline/markets.json   # input markets file
--threshold 0.78                               # cosine similarity cutoff for clustering
--min-confidence 0.70                          # minimum LLM confidence to keep a pair
--min-arb-edge 0.02                            # minimum edge (profit per $1) to report
--max-pairs-per-cluster 100                    # LLM call cap per cluster
```

### Step 3 — Poll for live arbitrage opportunities

```bash
python -m SemanticMatchPipeline.poller              # check once and exit
python -m SemanticMatchPipeline.poller --interval 30  # poll every 30 seconds
```

The poller reads `arb_pairs.json`, fetches live bid prices from each exchange, recomputes edge with fresh quotes, and prints any opportunity above the threshold.

Options:

```
--pairs SemanticMatchPipeline/arb_pairs.json   # path to arb_pairs.json
--interval 30                                  # seconds between polls (0 = run once)
--min-arb-edge 0.02
--min-confidence 0.70
```

## Project structure

```
UnifiedMarketPipeline/
  pipeline.py       — orchestrator: fetch → normalize → export/upsert
  adapters.py       — one adapter per exchange (Kalshi, Polymarket, Manifold, Metaculus)
  models.py         — UnifiedMarket schema
  vector_store.py   — Qdrant upsert logic
  markets.json      — last fetched market snapshot

SemanticMatchPipeline/
  pipeline.py       — orchestrator: cluster → classify → score → write outputs
  grouper.py        — Qdrant KNN + union-find clustering
  classifier.py     — Groq LLM pairwise relation classification
  arbitrage.py      — edge scoring per relation type
  models.py         — shared data models
  poller.py         — live price polling loop
  arb_pairs.json    — confirmed semantic pairs (written by pipeline, read by poller)
  opportunities.json — snapshot arb opportunities

qdrant_data/        — local Qdrant vector database
```
