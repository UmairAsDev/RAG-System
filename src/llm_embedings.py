from sentence_transformers import SentenceTransformer

def llm_embeddings(sentences):
    """
    Loads the SentenceTransformer model for generating embeddings.
    
    Returns:
        SentenceTransformer: A loaded embedding model if successful, or None if loading fails.
    """
    try:
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embed_model.encode(sentences)
        return embeddings
    except Exception as e:
        print(f"Error Loading loading embeddings {e}")
        return None
    

sentences = ["This is a test sentence.", "Here is another one."]
embeddings = llm_embeddings(sentences)

if embeddings is not None:
    print(embeddings)
else:
    print("Model loading failed or no embeddings generated.")