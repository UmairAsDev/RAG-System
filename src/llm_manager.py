from transformers import AutoModel

model_names  = ["gpt2","distilgpt2", "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-medium", "t5-small"]
generators = {model_name: AutoModel.from_pretrained(model_name) for model_name in model_names}

print(generators)
