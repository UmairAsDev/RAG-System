from langchain_community.vectorstores import FAISS
from embedding import embeddings
def get_vectorstore(text_chunks):
    embedding = embeddings()
    knowledge_base = FAISS.from_texts(text_chunks,embedding)
    return knowledge_base