from qdrant_client import QdrantClient
from llm_embedings import llm_embeddings

qdrant_client = QdrantClient(
    
)

print(qdrant_client.get_collections())