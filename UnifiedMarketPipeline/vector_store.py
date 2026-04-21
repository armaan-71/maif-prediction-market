import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "markets"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def setup_client() -> QdrantClient:
    """Initialize the local Qdrant client with FastEmbed enabled."""
    client = QdrantClient(path="qdrant_data")
    client.set_model(MODEL_NAME)
    return client


def upsert_markets(
    data: List[Dict[str, Any]],
    client: Optional[QdrantClient] = None,
    batch_size: int = 64,
    parallel: int = 0,
):
    """
    Core logic to insert/update market records in Qdrant.
    Can be called directly by the pipeline or other services.

    parallel=0 lets FastEmbed use all available CPU cores; parallel=1 is serial.
    """
    if client is None:
        client = setup_client()

    documents = []
    metadata = []
    ids = []

    for i, m in enumerate(data):
        # Extract text for embedding
        text = m.get("embedding_text") or m.get("text")
        if not text:
            # Fallback for raw JSON objects
            text = m.get("question", "") + " " + m.get("description", "")
            if not text.strip():
                continue

        documents.append(text)

        # Store all other useful fields in the payload (exclude text fields)
        payload = {k: v for k, v in m.items() if k not in ["embedding_text", "text"]}
        metadata.append(payload)

        # Consistent UUID based on the market UID or native_id for upserting
        unique_string = m.get("uid", m.get("native_id", str(i)))
        market_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        ids.append(market_id)

    if not documents:
        logger.warning("No valid markets with text found to upsert.")
        return

    logger.info(f"Upserting {len(documents)} markets to Qdrant collection '{COLLECTION_NAME}'...")
    logger.info(f"Embedding with batch_size={batch_size}, parallel={'all cores' if parallel == 0 else parallel}")

    # client.add() handles generating embeddings via FastEmbed and upserts
    client.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata,
        ids=ids,
        batch_size=batch_size,
        parallel=parallel,
    )
    logger.info(f"Successfully upserted {len(documents)} markets.")


def insert_markets_from_file(input_file: str):
    """CLI wrapper to read a JSON export and upsert it."""
    path = Path(input_file)
    if not path.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"Loading markets from {input_file}...")
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Expected JSON file to contain a list of market objects.")
        sys.exit(1)

    upsert_markets(data)


def search_markets(query: str, limit: int = 5):
    """Search for similar markets using FastEmbed."""
    client = setup_client()

    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        logger.error(
            f"Collection '{COLLECTION_NAME}' does not exist. Run an insert first."
        )
        sys.exit(1)

    logger.info(f"Searching for: '{query}'")
    results = client.query(
        collection_name=COLLECTION_NAME, query_text=query, limit=limit
    )

    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS FOR: '{query}'")
    print("=" * 80)

    for i, point in enumerate(results):
        score = point.score
        payload = point.metadata if hasattr(point, "metadata") else point.payload
        exchange = payload.get("exchange", "UNKNOWN").upper()
        question = payload.get("question", payload.get("title", "No Title"))
        price = payload.get("yes_price", "N/A")
        if isinstance(price, (float, int)):
            price = f"${price:.2f}"

        print(f"\n{i+1}. [Score: {score:.4f}] {exchange} | {question}")
        print(f"   Yes Price: {price}")
        print(f"   Native ID: {payload.get('native_id', 'N/A')}")
        if payload.get("url"):
            print(f"   URL: {payload.get('url')}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Qdrant Vector Store Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for markets")
    search_parser.add_argument("query", type=str, help="Search query string")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")

    # Insert command
    insert_parser = subparsers.add_parser("insert", help="Insert markets from JSON file")
    insert_parser.add_argument("file", type=str, help="Path to JSON file")

    args = parser.parse_args()

    if args.command == "search":
        search_markets(args.query, args.limit)
    elif args.command == "insert":
        insert_markets_from_file(args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

