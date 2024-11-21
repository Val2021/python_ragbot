
from sentence_transformers import SentenceTransformer
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def transform_data(data):
    if isinstance(data, float):
        return [data]

    elif isinstance(data, str):
        return [data]
    return data


def create_embedding(text):


    try:
        transformed_text = transform_data(text)
        embedding_ = model.encode(transformed_text)
        embedding = embedding_.tolist()

        logging.info(f"Generated embeddings for {len(embedding)} chunks.")
        # logging.info(f"First 3 embeddings: {embedding[:3]}")
        logging.info(f"Embedding type: {type(embedding)}")
        logging.info(f"First element type: {type(embedding[0]) if isinstance(embedding, list) else 'Not a list'}")

        return embedding
    except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return None
