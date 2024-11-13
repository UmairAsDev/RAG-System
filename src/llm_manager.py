from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

def load_model(model_name):
    try:
        if model_name == "Reformer":
            model = AutoModelForCausalLM.from_pretrained("google/reformer-enwik8")
            tokenizer = AutoTokenizer.from_pretrained("google/reformer-enwik8")
        elif model_name == "Blenderbot":
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
            tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        elif model_name == "T5":
            model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
            tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        elif model_name == "Bart":
            model = AutoModelForSeq2SeqLM.from_pretrained("lucadiliello/bart-small")
            tokenizer = AutoTokenizer.from_pretrained("lucadiliello/bart-small")
        elif model_name == "DistilGPT-2":
            model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
            tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        elif model_name == "DialoGPT":
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        elif model_name == "GPT2":
            tokenizer = AutoTokenizer.from_pretrained("ComCom/gpt2-small")
            model = AutoModelForCausalLM.from_pretrained("ComCom/gpt2-small")
        else:
            print(f"Model name: {model_name} is not recognized.")
            return None, None 

        return model, tokenizer

    except Exception as e:
        print(f"Cannot load model: {e}")
        return None, None


# Usage example
model, tokenizer = load_model("Bart")

if model is not None and tokenizer is not None:
    print("Model and tokenizer loaded successfully.")
    print(f"Model: {model}")
    print(f"Tokenizer: {tokenizer}")
else:
    print("Failed to load model or tokenizer.")
