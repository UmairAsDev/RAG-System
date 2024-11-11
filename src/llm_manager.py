from transformers import AutoModel
from llm_embedings import llm_embeddings
from vector_stores import get_vector_store
from vector_stores import add_documents

llm_embedding = llm_embeddings()

def llm_manager(select_model, llm_embedding, vector_db):
    select_model  = ["gpt2","distilgpt2", "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-medium", "t5-small"]
    generators = {model_name: AutoModel.from_pretrained(model_name) for model_name in select_model}
    
    


