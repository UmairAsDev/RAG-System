from sentence_transformers import SentenceTransformer
from document_processors import load_document

def embeddings():
    try:
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return embed_model
    except Exception as e:
        print(f"cannot create embeddings...{e}")
        return None        


        