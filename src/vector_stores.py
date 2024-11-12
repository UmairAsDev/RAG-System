from langchain_qdrant import QdrantVectorStore
import yaml
from llm_embedings import embeddings

embedding = embeddings()

def vector_database(docs):
    try:
        with open ('config/config.yaml', 'r') as file:
            secrets = yaml.safe_load(file)
            url = secrets["QDRANT_URL"]
            api_key = secrets["QDRANT_API"]
            
        qdrant_client = QdrantVectorStore.from_documents(
            docs,
            embedding,
            url=url,
            prefer_grpc = True,
            api_key=api_key,
            collection_name = "Rag_system",
        )
        return qdrant_client
    except Exception as e:
        print(f"Vector store is not created..{e}")
        return None
    

