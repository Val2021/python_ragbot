
from ingestion import load_dataset
from embedding import create_embedding
from db import store_embedding
from utils.extract_metadata import extract_title_page
from dotenv import load_dotenv, find_dotenv
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

def process_and_store_embeddings():
    """
    Process the dataset, generate embeddings for each chunk, and store them in the database.
    Each chunk is processed by generating embeddings and associating them with metadata (title and page).
    """
    logging.info("Start timing for loading the dataset")
    start_time = time.time()

    # Load and split the dataset into chunks
    texts = load_dataset()
    load_time = time.time() - start_time


    logging.info(f"Dataset loaded and split into chunks in {load_time:.2f} seconds.")

    # Start processing each chunk
    logging.info("Start processing each chunk")
    total_chunks = len(texts)

    last_titles_pages = None

    for i, chunk in enumerate(texts):
        # Log the processing of the chunk
        logging.info(f"Processing chunk {i + 1} of {total_chunks}.")

        try:
            # Measure the time for creating the embedding
            chunk_start_time = time.time()
            embedding = create_embedding(chunk)
            chunk_creation_time = time.time() - chunk_start_time
            logging.info(f"Embedding created for chunk {i + 1} in {chunk_creation_time:.2f} seconds.")

            # Extract the title and page metadata for the chunk
            current_titles_pages = extract_title_page([chunk])

            if not current_titles_pages:
                logging.warning(
                    f"Chunk {i + 1} has missing title/page. Using last known values."
                )
                current_titles_pages = last_titles_pages

            else:

                last_titles_pages = current_titles_pages

            # Measure the time for storing the embedding
            store_start_time = time.time()
            store_embedding(embedding, current_titles_pages)  # Store the embedding in Qdrant database
            store_time = time.time() - store_start_time
            logging.info(f"Embedding stored for chunk {i + 1} in {store_time:.2f} seconds.")


        except Exception as e:
            logging.error(f"Error processing chunk {i + 1}: {e}")

    # Log the total time for processing all chunks
    total_time = time.time() - start_time
    logging.info(f"Finished processing {total_chunks} chunks in {total_time:.2f} seconds.")

# Run the function to start processing
if __name__ == "__main__":
    process_and_store_embeddings()
