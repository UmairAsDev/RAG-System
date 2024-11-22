from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
import yaml
from llm_embedings import embeddings
import uuid
from datetime import datetime

# def vector_database(docs):
#     try:
#         with open(r'../config/qdrant.yaml', 'r') as file:
#             secrets = yaml.safe_load(file)
#             url = secrets["QDRANT_URL"]
#             api_key = secrets["QDRANT_API"]
            
#         for doc in docs:
#             doc.metadata = {
#                 "file_name": doc.metadata.get("file_name", f"Document_{uuid.uuid4().hex[:8]}"),
#                 "upload_time": doc.metadata.get("upload_time", str(datetime.now()))
#             }
        
#         embedding = embeddings()
#         qdrant_client = QdrantVectorStore.from_documents(
#             docs,
#             embedding,
#             url=url,
#             prefer_grpc=True,
#             api_key=api_key,
#             collection_name="Rag_system",
#         )
                
#         return qdrant_client
#     except yaml.YAMLError as yaml_error:
#         print(f"Error Loading Configuration: {yaml_error}")
#         return None
#     except Exception as e:
#         print(f"Vector store is not created..{e}")
#         return None


from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from llm_embedings import embeddings
import yaml
import uuid
from datetime import datetime

def alternative_text_splitter(doc):
    try:
        docs = []
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
        )

        # Check if document content is loaded correctly
        if not doc.page_content:
            raise ValueError("Document content is empty or not loaded correctly.")

        page_split = text_splitter.split_text(doc.page_content)
        for page_sub_split in page_split:
            page_no = doc.metadata.get('page', 0)
            metadata = {
                'source': doc.metadata.get('source', 'unknown'),
                'page_no': page_no + 1,
                'file_name': doc.metadata.get("file_name", f"Document_{uuid.uuid4().hex[:8]}"),
                'upload_time': doc.metadata.get("upload_time", str(datetime.now()))
            }
            doc_string = Document(page_content=page_sub_split, metadata=metadata)
            docs.append(doc_string)

        return docs
    except Exception as e:
        print(f"Failed to split the document: {e}")
        return None

def vector_database(docs):
    try:
        with open(r'../config/qdrant.yaml', 'r') as file:
            secrets = yaml.safe_load(file)
            url = secrets["QDRANT_URL"]
            api_key = secrets["QDRANT_API"]
            
        all_docs = []
        for doc in docs:
            split_docs = alternative_text_splitter(doc)
            all_docs.extend(split_docs)

        embedding = embeddings()
        qdrant_client = QdrantVectorStore.from_documents(
            all_docs,
            embedding,
            url=url,
            prefer_grpc=True,
            api_key=api_key,
            collection_name="Rag_system",
        )
                
        return qdrant_client
    except yaml.YAMLError as yaml_error:
        print(f"Error Loading Configuration: {yaml_error}")
        return None
    except Exception as e:
        print(f"Vector store is not created..{e}")
        return None
