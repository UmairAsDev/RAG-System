from langchain.chains.llm import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
from llm_manager import load_model
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
from document_processors import summarize_text
from langchain_core.runnables import RunnableSequence
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




from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

def conversation(model_name, history, qdrant_client):
    # Load the Hugging Face model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Initialize the pipeline
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe = HuggingFacePipeline(pipeline=hf_pipeline)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["history", "query", "retrieved_docs"],
        template=(
            "Conversation history:\n{history}\n\n"
            "Relevant documents:\n{retrieved_docs}\n\n"
            "User: {query}\nAI:"
        ),
    )

    memory = ConversationBufferWindowMemory(memory_key="history", k=5)

    def handle_query(user_query):
        try:
            # Check for casual greetings
            greetings = ["hi", "hello", "hey", "how are you"]
            if any(greeting in user_query.lower() for greeting in greetings):
                ai_response = "Hi there! How can I assist you today?"
                return {"response": ai_response, "retrieved_docs": ""}

            # Retrieve relevant documents for other queries
            retrieved_docs = cached_retrieval(user_query, qdrant_client)
            print(f"Retrieved {len(retrieved_docs)} documents")

            if not retrieved_docs:
                print("No documents retrieved.")
                retrieved_text = "No relevant documents found."
            else:
                retrieved_text = "\n".join([summarize_text(doc.page_content) for doc in retrieved_docs])

            prompt_input = prompt_template.format(history=history, query=user_query, retrieved_docs=retrieved_text)
            ai_response = pipe.invoke(prompt_input)

            print(f"AI Response: {ai_response}")

            return {"response": ai_response, "retrieved_docs": retrieved_text}
        except Exception as e:
            print(f"Error in handle_query: {e}")
            return {"response": "An error occurred while processing your query.", "retrieved_docs": ""}
    return handle_query
