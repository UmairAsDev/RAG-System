from transformers import AutoModel
from llm_embedings import llm_embeddings
from vector_stores import get_vector_store
def llm_manager(selected_model, llm_embeddings, get_vector_store):
    if selected_model == "gpt-neo":
        pass
        