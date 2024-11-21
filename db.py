from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, ScoredPoint
import logging
import os
import uuid
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Qdrant
qdrant_url = os.getenv("QDRANT_URL")
client = QdrantClient(url=qdrant_url)

# Initialize the collection for storing user data
collection_name = "python_data_doc"

# Vector configurations
vectors_config = VectorParams(size=384, distance="Cosine")  # Adjust size and distance metric as needed

# Create or recreate the collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=vectors_config
)


def store_embedding(embeddings, titles_pages):
    """
    Store multiple embeddings in the Qdrant database with metadata.
    Ensure no duplicate embeddings for the same title and page.

    Args:
        embeddings (list): A list of embedding vectors to store.
        titles_pages (list): A list of dictionaries with "title" and "page" keys for metadata.

    Returns:
        list[str]: A list of IDs of the stored points.
    """
    points = []  # List to store all points for batch insertion
    stored_ids = []  # List to keep track of stored point IDs

    for embedding, metadata in zip(embeddings, titles_pages):
        # Check if the point already exists in the collection
        existing_point = client.search(collection_name=collection_name,
            query_vector=embedding,
            limit=1,
        )

        logging.info(f"Point existing_point '{existing_point}'")
        logging.info(f"Point existing_point type '{type(existing_point)}'")


        point_id = None

        if existing_point:
            if isinstance(existing_point, list) and existing_point:
                for point in existing_point:
                    if point.payload["title"] == metadata["title"] and point.payload["page"] == metadata["page"]:
                        logging.info(f"Point with title '{metadata['title']}' and page '{metadata['page']}' already exists, using existing ID.")
                        point_id = point.id
                        break

        if not point_id:
            point_id = str(uuid.uuid4())  # Always use a valid string ID
            logging.info(f"Point with title '{metadata['title']}' and page '{metadata['page']}' does not exist, creating new ID.")


        if not isinstance(point_id, (str, int)):
            logging.error(f"Generated point_id is invalid: {point_id}")
            continue

        # Create the point with the ID (either new or existing)
        point = {
            "id": point_id,
            "vector": embedding,
            "payload": metadata  # Add metadata if provided
        }

        points.append(point)
        stored_ids.append(point_id)

    if points:
        try:
            client.upsert(collection_name=collection_name, points=points)
            logging.info(f"{len(points)} embeddings stored successfully.")
            return stored_ids
        except Exception as e:
            logging.error(f"Error storing embedding in Qdrant: {e}")
            logging.error(f"Full error traceback: {traceback.format_exc()}")
            return None
    else:
        logging.info("No new embeddings to store.")
        return []
