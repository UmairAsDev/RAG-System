from langchain_huggingface import HuggingFaceEmbeddings

def embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        print(f"Embeddings have created sucessfully...{e}")
        



