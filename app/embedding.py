
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from sentence_transformers import SentenceTransformer

def create_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    logging.info(f"Generated embeddings for {len(embedding)} chunks.")
    return embedding
