import streamlit as st
from document_processors import load_document, document_splitters
from vector_stores import vector_database
from conversation_chain import conversation

def main():
    st.set_page_config(page_title="Conversational AI", page_icon="🤖", layout="wide")
    st.header("Conversational AI")

    try:
        # File upload
        uploaded_files = st.sidebar.file_uploader(
            "Upload Your Document",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True
        )

        # Initialize session state variables
        if "history" not in st.session_state:
            st.session_state.history = ""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "qdrant_client" not in st.session_state:
            st.session_state.qdrant_client = None

        # Process uploaded files
        if uploaded_files:
            document = load_document(uploaded_files)
            if document:
                split_docs = document_splitters(document)
                if split_docs:
                    qdrant_client = vector_database(split_docs)
                    if qdrant_client:
                        st.session_state.qdrant_client = qdrant_client
                        st.success("Document successfully loaded and stored in the database.")
                    else:
                        st.error("Failed to initialize the vector database.")
                else:
                    st.error("Failed to split the document.")
            else:
                st.error("Failed to load document.")

        # Model selection
        model_name = st.sidebar.selectbox(
            "Select LLM",
            ["Reformer", "Blenderbot", "T5", "Bart", "DistilGPT-2", "DialoGPT", "GPT2"],
        )

        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input widget
        user_query = st.chat_input("Type your message here...")

        if user_query:
        # Display user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
               st.write(user_query)

            # Process user query
            if st.session_state.qdrant_client:
                conversation_handler = conversation(
                    model_name, split_docs, st.session_state.history, st.session_state.qdrant_client
                )
                result = conversation_handler(user_query)
                ai_response = result["response"]

                # Display AI response
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.write(ai_response)

                # Update history
                st.session_state.history += f"\nUser: {user_query}\nAI: {ai_response}"
            else:
                st.warning("Please upload and process documents before starting the conversation.")


    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
