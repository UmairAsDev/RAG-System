from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import DistilBertModel


embed_model = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

qdrant_client = QdrantClient(
    
)

print(qdrant_client.get_collections())