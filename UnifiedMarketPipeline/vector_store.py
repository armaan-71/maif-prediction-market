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
DEFAULT_QDRANT_PATH = "qdrant_data"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def setup_client(
    path: Optional[str] = None,
    collection: Optional[str] = None,
) -> QdrantClient:
    """Initialize the local Qdrant client with FastEmbed enabled.

    Args:
        path: Qdrant storage path. Use ":memory:" for an ephemeral in-process DB,
              or a directory path for a persistent local DB.
              Defaults to DEFAULT_QDRANT_PATH ("qdrant_data").
        collection: Collection name to use. Stashed on the client so downstream
                    helpers can resolve it without an extra argument.
    """
    resolved_path = path if path is not None else DEFAULT_QDRANT_PATH
    if resolved_path == ":memory:":
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(path=resolved_path)
    client.set_model(MODEL_NAME)
    # Stash so helpers can resolve the collection without threading it everywhere
    client._collection_name = collection or COLLECTION_NAME  # type: ignore[attr-defined]
    return client


def _resolve_collection(client: QdrantClient, collection: Optional[str]) -> str:
    if collection:
        return collection
    return getattr(client, "_collection_name", COLLECTION_NAME)


def upsert_markets(
    data: List[Dict[str, Any]],
    client: Optional[QdrantClient] = None,
    batch_size: int = 64,
    parallel: int = 0,
    collection: Optional[str] = None,
):
    """
    Core logic to insert/update market records in Qdrant.
    Can be called directly by the pipeline or other services.

    parallel=0 lets FastEmbed use all available CPU cores; parallel=1 is serial.

    Args:
        collection: Override the collection name. Falls back to client._collection_name,
                    then the module-level COLLECTION_NAME constant.
    """
    if client is None:
        client = setup_client()

    col = _resolve_collection(client, collection)

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
        market_id = str(uuid.uuid5(uuid.NAMESPACE_URL, unique_string))
        ids.append(market_id)

    if not documents:
        logger.warning("No valid markets with text found to upsert.")
        return

    logger.info(f"Upserting {len(documents)} markets to Qdrant collection '{col}'...")
    logger.info(f"Embedding with batch_size={batch_size}, parallel={'all cores' if parallel == 0 else parallel}")

    # client.add() handles generating embeddings via FastEmbed and upserts
    client.add(
        collection_name=col,
        documents=documents,
        metadata=metadata,
        ids=ids,
        batch_size=batch_size,
        parallel=parallel,
    )
    logger.info(f"Successfully upserted {len(documents)} markets.")


def insert_markets_from_file(
    input_file: str,
    path: Optional[str] = None,
    collection: Optional[str] = None,
):
    """CLI wrapper to read a JSON export and upsert it."""
    file_path = Path(input_file)
    if not file_path.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"Loading markets from {input_file}...")
    with open(file_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Expected JSON file to contain a list of market objects.")
        sys.exit(1)

    client = setup_client(path=path, collection=collection)
    upsert_markets(data, client=client, collection=collection)


def search_markets(
    query: str,
    limit: int = 5,
    path: Optional[str] = None,
    collection: Optional[str] = None,
):
    """Search for similar markets using FastEmbed."""
    client = setup_client(path=path, collection=collection)
    col = _resolve_collection(client, collection)

    try:
        client.get_collection(col)
    except Exception:
        logger.error(
            f"Collection '{col}' does not exist. Run an insert first."
        )
        sys.exit(1)

    logger.info(f"Searching for: '{query}'")
    results = client.query(
        collection_name=col, query_text=query, limit=limit
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
    search_parser.add_argument("--qdrant-path", type=str, default=None,
                               help="Qdrant storage path (default: qdrant_data)")
    search_parser.add_argument("--collection", type=str, default=None,
                               help="Collection name (default: markets)")

    # Insert command
    insert_parser = subparsers.add_parser("insert", help="Insert markets from JSON file")
    insert_parser.add_argument("file", type=str, help="Path to JSON file")
    insert_parser.add_argument("--qdrant-path", type=str, default=None,
                               help="Qdrant storage path (default: qdrant_data)")
    insert_parser.add_argument("--collection", type=str, default=None,
                               help="Collection name (default: markets)")

    args = parser.parse_args()

    if args.command == "search":
        search_markets(args.query, args.limit,
                       path=args.qdrant_path, collection=args.collection)
    elif args.command == "insert":
        insert_markets_from_file(args.file,
                                 path=args.qdrant_path, collection=args.collection)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

