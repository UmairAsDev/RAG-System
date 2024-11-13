from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

def embeddings():
    try:
       embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
       return embeddings_model
    except Exception as e:
        print(f"cannot create embeddings...{e}")
        return None        


        