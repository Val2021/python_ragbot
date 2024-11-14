from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, ScoredPoint
import logging
import os
import uuid

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


def store_embedding(embedding, metadata=None):
    """
    Store an embedding in the Qdrant database with optional metadata.

    Args:
        embedding (list): The embedding vector to store.
        metadata (dict): Additional metadata to store with the embedding (e.g., chunk ID, text).

    Returns:
        str: The ID of the stored point.
    """
    point_id = str(uuid.uuid4())  # Generate a unique ID for the point

    point = {
        "id": point_id,
        "vector": embedding,
        "payload": metadata or {}  # Add metadata if provided
    }

    try:
        client.upsert(collection_name=collection_name, points=[point])
        logging.info(f"Embedding stored successfully with ID: {point_id}")
        return point_id
    except Exception as e:
        logging.error(f"Error storing embedding in Qdrant: {e}")
        return None


def retrieve_embeddings(query_vector, top_k=5):
    """
    Retrieve the most relevant embeddings from Qdrant based on a query vector.

    Args:
        query_vector (list): The embedding vector representing the query.
        top_k (int): The number of top results to return.

    Returns:
        list[ScoredPoint]: A list of the most relevant points with scores and metadata.
    """
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        logging.info(f"Retrieved {len(results)} results from the database.")
        return results
    except Exception as e:
        logging.error(f"Error retrieving embeddings from Qdrant: {e}")
        return []
