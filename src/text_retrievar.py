from langchain.chains.llm import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
from llm_manager import load_model
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import torch

def retrieval_engine(model_name):
    model, tokenizer = load_model(model_name)
    
    if model is None or tokenizer is None:
        print(f"Failed to load mode..")
        return
    
    
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens = 100,
        max_length=200 
    )

    pipe = HuggingFacePipeline(pipeline=hf_pipeline)
    
    prompt_template = PromptTemplate(
        input_variables= ["history", "query"],
        template= "conversation history:\n{history}\n\nUser: {query}\nAI:"
    )
    
    memory = ConversationBufferMemory(memory_key = "history")
    
    conversation_chain = LLMChain(
        llm=pipe,
        prompt=prompt_template,
        memory=memory
    )
    
    while True:
         query = input("User: ")
         if query.lower() in ["exit", "quit"]:
             print("Ending the Conversation")
             break
         response = conversation_chain.run(query=query)
         print("AI:", response)


# retrieval_engine("GPT2")
        