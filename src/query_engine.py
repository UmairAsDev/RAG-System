import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader

def load_document(uploaded_file):
    try:
        # Save the uploaded file temporarily on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load document based on file type
        if temp_file_path.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
            content = loader.load()
        elif temp_file_path.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
            content = loader.load()
        elif temp_file_path.endswith(".csv"):
            loader = CSVLoader(temp_file_path)
            content = loader.load()
        elif temp_file_path.endswith(".txt"):
            loader = TextLoader(temp_file_path)
            content = loader.load()
        else:
            st.error("Unsupported document format.")
            return None

        return content

    except Exception as e:
        st.error(f"Cannot load document... {e}")
        return None

# Streamlit app interface
uploaded_file = st.file_uploader("Upload Your Document", type=["pdf", "txt", "docx", "csv"])

if uploaded_file is not None:
    document = load_document(uploaded_file)
    if document is not None:
        st.write("Document content loaded successfully.")
        st.write(document)
