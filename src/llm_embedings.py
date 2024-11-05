from sentence_transformers import SentenceTransformer
from transformers import DistilBertModel
import torch
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = DistilBertModel("distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation = "eager")


