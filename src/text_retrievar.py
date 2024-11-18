from langchain.chains.llm import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
from llm_manager import load_model
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from transformers import pipeline
import torch
from vector_stores import vector_database
import streamlit as st

# def truncate_text(text, max_tokens=500):
#     words = text.split()
#     return " ".join(words[:max_tokens]) if len(words) > max_tokens else text



# def get_user_input():
#     return st.text_input("Your query:") if is_streamlit else input("User: ")

# def display_response(response):
#     if is_streamlit:
#         st.write("AI:", response)
#     else:
#         print("AI:", response)




    # pipe = HuggingFacePipeline(pipeline=hf_pipeline)

    # prompt_template = PromptTemplate(
    #     input_variables=["history", "query", "retrieved_docs"],
    #     template="conversation history:\n{history}\n\nRelevant documents:\n{retrieved_docs}\n\nUser: {query}\nAI:",
    # )

    # memory = ConversationBufferWindowMemory(memory_key="history", k=5)
    # conversation_chain = LLMChain(llm=pipe, prompt=prompt_template, memory=memory)

    # qdrant_client = vector_database(docs)
    # if qdrant_client is None:
    #     print("Vector database not initialized. Exiting.")
    #     return

    # print("Start chatting with the AI! Type 'exit' or 'quit' to end the conversation.")
    # while True:
    #     query = get_user_input()
    #     if query.lower() in ["exit", "quit", "bye", "stop"]:
    #         print("Ending the conversation. Goodbye!")
    #         break

    #     retrieved_docs = cached_retrieval(query, qdrant_client)
    #     retrieved_text = "\n".join([truncate_text(doc.page_content) for doc in retrieved_docs])
    #     display_response(conversation_chain.run(query=query, retrieved_docs=retrieved_text))


def conversation(model_name, docs):
    model, tokenizer = load_model(model_name)
    if model is None or tokenizer is None:
        print("Failed to load model. Please check the model name and try again.")
        return

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=100,
        max_length=200,
    )
    
    response = hf_pipeline(docs, max_new_tokens=512)
    print(response[0]['generated_text'][-1]['content'])
    


docs = [
    {"role": "system", "content": "You are a sassy, Pakistani who like to brag and show off their wealth."},
    {"role": "user", "content": "Hey, whats the best thing to eat in pakistan"}
]


response = conversation("DialoGPT" , docs)

print(response)