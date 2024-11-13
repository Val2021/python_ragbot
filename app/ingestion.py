import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv(find_dotenv(), override=True)

def load_dataset():
    loader = PyPDFLoader('dataset/python_doc_3.13.pdf')
    data = loader.load()
    logging.info(f"PDF loaded successfully. Total of {len(data)} pages extracted.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    texts = text_splitter.split_documents(data)
    logging.info(f"Text splitting completed. Total of {len(texts)} chunks generated.")
    
    return texts
