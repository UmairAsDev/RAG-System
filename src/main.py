import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_embedings import llm_embeddings
from vector_stores import get_vector_store, add_documents
import os

# Define the LLM manager (conversation manager)
def llm_manager(select_model, input_text, selected_document):
    try:
        model_names = [
            "gpt2", 
            "distilgpt2", 
            "facebook/blenderbot-400M-distill", 
            "microsoft/DialoGPT-medium", 
            "t5-small"
        ]
        
        # Load the models and tokenizers dynamically
        generators = {name: (
            AutoModelForCausalLM.from_pretrained(name), 
            AutoTokenizer.from_pretrained(name)
        ) for name in model_names}
        
        # Get vector store (Qdrant client)
        qdrant_client = get_vector_store()
        
        if select_model in generators:
            model, tokenizer = generators[select_model]
            
            # Tokenize the input text and generate model output
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Optionally, store the conversation context or input as an embedding in the vector store
            llm_embedding = llm_embeddings([input_text, generated_text])
            
            # Only add to vector store if embeddings are successfully generated
            if llm_embedding is not None:
                add_documents(qdrant_client, [selected_document, input_text, generated_text])

            return generated_text

        else:
            raise ValueError(f"Unsupported model selected: {select_model}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Streamlit UI

st.title("Conversation with Documents and Models")

# Document Upload
uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for file in uploaded_files:
        document_text = file.read().decode("utf-8")  # Assuming the document is text-based
        documents.append({"name": file.name, "text": document_text})
    
    # Show the uploaded documents in a dropdown for the user to select from
    document_names = [doc["name"] for doc in documents]
    selected_document_name = st.selectbox("Select a document for conversation", document_names)

    # Get the selected document text
    selected_document = next(doc for doc in documents if doc["name"] == selected_document_name)["text"]

    # Model and Embedding Selection
    model_choice = st.selectbox("Select a conversation model", ["gpt2", "distilgpt2", "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-medium", "t5-small"])
    
    input_text = st.text_input("Enter your message")

    if st.button("Start Conversation"):
        if input_text:
            response = llm_manager(model_choice, input_text, selected_document)
            st.write(f"Model's response: {response}")
        else:
            st.write("Please enter a message to start the conversation.")
