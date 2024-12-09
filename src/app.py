import streamlit as st
from dotenv import load_dotenv
from pre_process import get_files_text, get_text_chunks
from vector_db import get_vectorstore
from retrieval import get_conversation_chain, handel_userinput
import os

openapi_key = os.environ.get("OPENAI_API_KEY")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.header("DocumentGPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf'],accept_multiple_files=True)
        openai_api_key = openapi_key
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        st.write("file chunks created...")
        # create vetore stores
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vectore Store Created...")
         # create conversation chain
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) #for openAI

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)

if __name__ == '__main__':
    main()