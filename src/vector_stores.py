from qdrant_client import QdrantClient
from llm_embedings import llm_embeddings
import yaml

with open(r'config\config.yaml', 'r') as file:
    secrets = yaml.safe_load(file)

qdrant_url = secrets["QDRANT_URL"]
qdrant_api = secrets["QDRANT_API"]
collection_name = "Multi_Rag_System"

qdrant_client = QdrantClient(
    embed_model = llm_embeddings,
    url = qdrant_url,
    prefer_grpc=True,
    api_key=qdrant_api,
    collection_name = collection_name,
)

