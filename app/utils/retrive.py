
import os
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.embeddings.base import Embeddings
from langchain_qdrant import QdrantVectorStore
from app.embedding import create_embedding


import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Qdrant
qdrant_url = os.getenv("QDRANT_URL")
client = QdrantClient(url=qdrant_url)
collection_name = "python_data_doc"


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv(), override=True)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]


logger = logging.getLogger()
model = ChatGroq(model="llama3-8b-8192")


class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [create_embedding(text) for text in texts]

    def embed_query(self, text):
        return create_embedding(text)

embeddings = CustomEmbeddings()

retriever = QdrantVectorStore.from_existing_collection(
    embeddings=embeddings,
    collection_name=collection_name,
    client=client,
)

retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),


     ]
)
contextualize_q_llm = model.with_config(tags=["contextualize_q_llm"])
history_aware_retriever = create_history_aware_retriever(model,retriever,contextualize_q_prompt)
# Configurando a chain de QA com LangChain

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

question1 = "what is python?"
message1= rag_chain.invoke({"input": question1, "chat_history": chat_history})
print(message1)
