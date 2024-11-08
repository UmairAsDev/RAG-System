from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance
from llm_embedings import llm_embeddings
import yaml
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams



def get_vector_store():
    try:
        with open(r'config/config.yaml', 'r') as file:
            secrets = yaml.safe_load(file)

        qdrant_url = secrets["QDRANT_URL"]
        qdrant_api = secrets["QDRANT_API"]
        collection_name = "Multi_Rag_System"

        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api,
            prefer_grpc=True
        )

        try:
            qdrant_client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            print(f"Collection '{collection_name}' does not exist. Creating it now.")
            vector_params = VectorParams(
                size=384,
                distance=Distance.COSINE 
            )
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params 
            )
            print(f"Collection '{collection_name}' created successfully.")

        return qdrant_client

    except Exception as e:
        print(f"Vector store can't be created... {e}")
        return None



def add_documents(qdrant_client, documents , collection_name= "Multi_Rag_System"):
    
    try:
        embeddings = llm_embeddings(documents)
        
        if embeddings is None:
            print("Embeddngs  generation failed.")
            return False
        
        points = [
            PointStruct(id= 'i', vector=embedding, payload={"text" : document})
            for i , (embedding, document) in enumerate(zip(embeddings, documents))
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print("Document added successfully")
        return True
    
    except Exception as e:
        print(f"Failed to add documents  to Qdrant")
        return False
    
    
  
qdrant_client = get_vector_store()

documents = ["Sample text one", "Sample text two", "Sample text three"]

if qdrant_client:
    add_documents(qdrant_client, documents)
    
        