from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_embedings import llm_embeddings
from vector_stores import get_vector_store, add_documents

def llm_manager(select_model, input_text):
    try:
        model_names = [
            "gpt2", 
            "distilgpt2", 
            "facebook/blenderbot-400M-distill", 
            "microsoft/DialoGPT-medium", 
            "t5-small"
        ]
        
        generators = {name: (
            AutoModelForCausalLM.from_pretrained(name), 
            AutoTokenizer.from_pretrained(name)
        ) for name in model_names}
        

        qdrant_client = get_vector_store()
        
        if select_model in generators:
            model, tokenizer = generators[select_model]
            
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            llm_embedding = llm_embeddings([input_text, generated_text])
            
            if llm_embedding is not None:
                add_documents(qdrant_client, [input_text, generated_text])

            return generated_text

        else:
            raise ValueError(f"Unsupported model selected: {select_model}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    
    select_model = "gpt2" 
    input_text = "Hello, how are you?"
    
    result = llm_manager(select_model, input_text)
    if result:
        print("Model response:", result)
    else:
        print("Something went wrong.")
