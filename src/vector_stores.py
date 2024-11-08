from langchain_qdrant import QdrantVectorStore
from llm_embedings import llm_embeddings
import yaml

def get_vector_store():
    try:
        with open(r'config\config.yaml', 'r') as file:
            secrets = yaml.safe_load(file)

        qdrant_url = secrets["QDRANT_URL"]
        qdrant_api = secrets["QDRANT_API"]
        collection_name = "Multi_Rag_System"

        qdrant_client = QdrantVectorStore.from_documents(
            embed_model = llm_embeddings,
            url = qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api,
            collection_name = collection_name,
        )
        return qdrant_client
    except Exception as e:
        print(f"Vector store can't be created...{e}")
        return None

