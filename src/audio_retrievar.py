from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
import openai
import re
import os

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


import spacy

nlp = spacy.load('en_core_web_sm')
loader = PyPDFLoader('National AI Policy Consultation.pdf')

pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

from langchain.docstore.document import Document

# Ensure doc_list is initialized
doc_list = []

# Split each page and convert into Document objects
for page in pages:
    pg_split = text_splitter.split_text(page.page_content)

    # Loop through each split sec   tion of the page
    for pg_sub_split in pg_split:
        # Handle page number safely
        page_no = page.metadata.get('page', 0)  # Use 0 if page number isn't found
        metadata = {'source': 'National AI Policy', 'page_no': page_no + 1}

        # Create a Document object with the split content and metadata
        doc_string = Document(page_content=pg_sub_split, metadata=metadata)

        # Append the Document object to the doc_list
        doc_list.append(doc_string)


qdrant_url = "https://db068bf5-9309-4b11-9693-e232394b69d7.europe-west3-0.gcp.cloud.qdrant.io"
qdrant_key = "hjbXGKt-BJWS_Km9ZKGzPSoRkOp-09399us22wxBaMyo6Wx19DgAJQ"
collection_name = "AI Policy"


qdrant = QdrantVectorStore.from_documents(
    doc_list,
    embed_model,
    url = qdrant_url,
    api_key = qdrant_key,
    collection_name = collection_name
)


query = "What is the current Ai policy for students?"
results = qdrant.similarity_search(query, k=5)

import getpass
import os
import time

from pinecone import Pinecone, ServerlessSpec

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

import time

index_name = "langchain-test-index"  

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore.from_documents(doc_list, embed_model, index_name=index_name )

doc_ss = qdrant.similarity_search(query, k=5)
docs = qdrant.similarity_search(
    query,
    k=3,
    filter= {
        "must": [
            {
                "key": "page",
                "match": {
                    "value" : 0
                }
            }
        ]
    }
)

from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever

texts = [doc.page_content for doc in doc_list]
svm_retrievar = SVMRetriever.from_texts(texts, embed_model)
tfidf_retrievar = TFIDFRetriever.from_texts(texts)

question = "why national ai policy is so important?"
doc_svm = svm_retrievar.get_relevant_documents(question)
doc_svm[0]

question = "Why we need ai implementation on national level"
doc_tfidf = tfidf_retrievar.get_relevant_documents(question)
doc_tfidf[0]



