from langchain.chains.llm import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
from llm_manager import load_model
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from transformers import pipeline
import torch
from vector_stores import vector_database
import streamlit as st

# Define the function to retrieve documents from the vector store based on the query
def cached_retrieval(query, qdrant_client):
    try:
        # Perform a retrieval from the vector database using the query
        # For simplicity, assuming qdrant_client has a method 'retrieve' that retrieves the documents
        retrieved_docs = qdrant_client.similarity_search(query, k=5)  # Adjust `k` as needed
        return retrieved_docs
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        return []

def truncate_text(text, max_tokens=500):
    words = text.split()
    return " ".join(words[:max_tokens]) if len(words) > max_tokens else text

def get_user_input(is_streamlit):
    return st.text_input("Your query:") if is_streamlit else input("User: ")




def display_response(response, is_streamlit):
    if is_streamlit:
        st.write("AI:", response)
    else:
        print("AI:", response)



def conversation(model_name, docs, history, qdrant_client):
    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)
    if model is None or tokenizer is None:
        raise ValueError("Failed to load model. Please check the model name and try again.")

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=10000,
        max_length=200,
    )
    pipe = HuggingFacePipeline(pipeline=hf_pipeline)

    # Prompt template for the conversation
    prompt_template = PromptTemplate(
        input_variables=["history", "query", "retrieved_docs"],
        template="conversation history:\n{history}\n\nRelevant documents:\n{retrieved_docs}\n\nUser: {query}\nAI:",
    )

    memory = ConversationBufferWindowMemory(memory_key="history", k=5)
    conversation_chain = LLMChain(llm=pipe, prompt=prompt_template, memory=memory)

    def handle_query(user_query):
        try:
            # Retrieve relevant documents
            retrieved_docs = cached_retrieval(user_query, qdrant_client)
            print(f"Retrieved {len(retrieved_docs)} documents")

            if not retrieved_docs:
                raise ValueError("No documents retrieved")

            retrieved_text = "\n".join([truncate_text(doc.page_content) for doc in retrieved_docs])

            # Generate response from the conversation chain
            ai_response = conversation_chain.run(
                history=history,
                query=user_query,
                retrieved_docs=retrieved_text,
            )

            return {"response": ai_response, "retrieved_docs": retrieved_text}
        except Exception as e:
            print(f"Error in handle_query: {e}")
            return {"response": "An error occurred while processing your query.", "retrieved_docs": ""}

    return handle_query

