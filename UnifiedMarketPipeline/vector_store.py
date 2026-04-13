import argparse
import json
import logging
import sys
from pathlib import Path
from qdrant_client import QdrantClient, models

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


def insert_markets(input_file: str):
    """Read a JSON export of markets and insert them into Qdrant."""
    client = setup_client()
    path = Path(input_file)

    if not path.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"Loading markets from {input_file}...")

    # Read the pipeline exported JSON (expects a list of dicts)
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Expected JSON file to contain a list of market objects.")
        sys.exit(1)

    # Qdrant's .add() expects parallel lists of documents (text to embed),
    # metadata (payload dicts), and ids.
    documents = []
    metadata = []
    ids = []

    import uuid

    for i, m in enumerate(data):
        text = m.get("embedding_text") or m.get("text")
        if not text:
            # Fallback if raw JSON object was provided without embedding_text
            text = m.get("question", "") + " " + m.get("description", "")
            if not text.strip():
                continue

        documents.append(text)

        # Store all other useful fields in the payload
        payload = {k: v for k, v in m.items() if k not in ["embedding_text", "text"]}
        metadata.append(payload)

        # Consistent UUID based on the market UID or native_id
        unique_string = m.get("uid", m.get("native_id", str(i)))
        market_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        ids.append(market_id)

    if not documents:
        logger.warning("No valid markets with text found to insert.")
        return

    logger.info(
        f"Generating FastEmbed vectors and upserting {len(documents)} markets..."
    )

    # client.add() handles generating embeddings via FastEmbed,
    client.add(
        collection_name=COLLECTION_NAME, documents=documents, metadata=metadata, ids=ids
    )

    logger.info(
        f"Successfully inserted {len(documents)} markets into collection '{COLLECTION_NAME}'."
    )


def search_markets(query: str, limit: int = 5):
    """Search for similar markets using FastEmbed."""
    client = setup_client()

    # Check if collection exists
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        logger.error(
            f"Collection '{COLLECTION_NAME}' does not exist. Please run 'insert' first."
        )
        sys.exit(1)

    logger.info(f"Searching for: '{query}'")

    # client.query() automatically embeds the query string and performs vector search
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
        if isinstance(price, float):
            price = f"${price:.2f}"

        print(f"\n{i+1}. [Score: {score:.4f}] {exchange} | {question}")
        print(f"   Yes Price: {price}")
        print(f"   Native ID: {payload.get('native_id', 'N/A')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Insert data
    # insert_markets("test_markets.json")

    # Search data
    search_markets("Will Trump")
