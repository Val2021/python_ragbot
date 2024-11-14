import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv(find_dotenv(), override=True)


loader = PyPDFLoader('dataset/python_doc_3.13.pdf')
data = loader.load()


def clean_text(text):
        # Fix concatenated words like "Codingstandards" -> "Coding standards"
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)
        return text

def load_dataset():
    combined_text = " ".join([clean_text(page.page_content) for page in data])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=300,
        separators=["\n", " "]
    )

    chunks = text_splitter.split_text(combined_text)

    logging.info(f"Total chunks generated: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):
        logging.info(f"Chunk {i+1}: {chunk[:500]}...\n")

    return chunks
