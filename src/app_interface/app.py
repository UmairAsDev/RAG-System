# import streamlit as st
# from pages import home, data_upload, data_analysis, query_interface, model_selection, history, about

# st.set_page_config("RAG-PRO")

# PAGES = {
#     "Home": home,
#     "Data Upload": data_upload,
#     "Data Analysis": data_analysis,
#     "Query Interface": query_interface,
#     "Model Selection & Settings": model_selection,
#     "History & Logs": history,
#     "About": about
# }

# st.sidebar.title("Navigation")
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# # Display the selected page
# page = PAGES[selection]
# page.app()
import logging

logging.basicConfig(level=logging.INFO)

import sys
import os

# Add the src directory to sys.path to allow imports from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import streamlit as st
from llm_manager import load_model
from vector_stores import store_document_embeddings, retrieve_similar_embeddings
from llm_embedings import generate_embeddings
from query_engine import chat_with_model, update_chat_history

st.sidebar.title("Document & Model Selection")
uploaded_files = st.sidebar.file_uploader("Upload Documents", type=['pdf', 'txt'], accept_multiple_files=True)
selected_model = st.sidebar.selectbox("Select Conversation Model", ["DistilGPT", "TinyBERT", "gpt2", "t5-small", "Dialogflow"])

llm_model = load_model(selected_model)



if uploaded_files:
    for file in uploaded_files:
        content = None
        for encoding in ["utf-8", "ISO-8859-1", "utf-16"]:
            try:
                # Attempt to read with different encodings
                content = file.read().decode(encoding)
                logging.info(f"File {file.name} decoded successfully with {encoding} encoding.")
                break  # Exit the loop if successful
            except UnicodeDecodeError:
                # If this encoding fails, try the next one
                logging.warning(f"Failed to decode {file.name} with {encoding}. Trying next encoding.")
                continue
            except Exception as e:
                # General exception handling for any other errors
                logging.error(f"Error processing file {file.name}: {e}")
                continue

        # If no valid encoding was found, skip the file
        if content is None:
            logging.warning(f"Skipping file {file.name} after trying all encodings.")
            continue

        # Generate embeddings and store them
        try:
            embeddings = generate_embeddings(content)
            store_document_embeddings(embeddings, document_id=file.name)
            logging.info(f"Embeddings for {file.name} stored successfully.")
        except Exception as e:
            logging.error(f"Error generating embeddings for {file.name}: {e}")


st.title("Conversational AI with RAG")
user_input = st.text_input("Enter your message:")
if user_input:
    query_embedding = generate_embeddings(user_input)
    retrieved_docs = retrieve_similar_embeddings(query_embedding)
    response = chat_with_model(llm_model, user_input, retrieved_docs)
    update_chat_history(user_input, response)
    st.write(f"Bot: {response}")
