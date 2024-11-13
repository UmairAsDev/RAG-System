from langchain_qdrant import QdrantVectorStore
import yaml
from llm_embedings import embeddings
from document_processors import document_splitters


def vector_database(docs):
    try:
        with open ('config/config.yaml', 'r') as file:
            secrets = yaml.safe_load(file)
            url = secrets["QDRANT_URL"]
            api_key = secrets["QDRANT_API"]
            
        docData = document_splitters(docs)
        if docData or not isinstance(docData, list):
            raise ValueError (f"Document Processing returned empty or invalid format.")
        
        embedding = embeddings()

          
        qdrant_client = QdrantVectorStore.from_documents(
            docData,
            embedding,
            url=url,
            prefer_grpc = True,
            api_key=api_key,
            collection_name = "Rag_system",
        )
        return qdrant_client
    except yaml.YAMLError as yaml_error:
        print(f"Error Loading Configuration:{yaml_error}")
        return None
    except Exception as e:
        print(f"Vector store is not created..{e}")
        return None



