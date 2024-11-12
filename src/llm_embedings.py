from langchain_huggingface import HuggingFaceEmbeddings

def embeddings():
    try:
        embed_model = HuggingFaceEmbeddings(model = "sentence-tranformers/all-mpnet-base-v2")
        return embed_model
    except Exception as e:
        print(f"cannot create embeddings...{e}")
        return None        
        
    