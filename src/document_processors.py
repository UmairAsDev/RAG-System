from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import streamlit as st
from langchain.schema import Document
def load_document(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            file_path = temp_file.name
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            content = loader.load()
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            content = loader.load()
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
            content = loader.load()
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
            content = loader.load()
        else:
            st.error("Unsupported document format.")
            return None

        return content

    except Exception as e:
        st.error(f"Cannot load document... {e}")
        return None
    
   

 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Ensure Document is correctly imported

def document_splitters(document):
    """
    Splits a document into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        document (list): A list of objects, each with `page_content` (str) and `metadata` (dict).

    Returns:
        list: A list of Document objects with split content and updated metadata.
    """
    try:
        # Input validation
        if not isinstance(document, list):
            raise TypeError("Input document must be a list of objects with 'page_content' and 'metadata'.")
        
        docs = []
        MARKDOWN_SEPARATORS = [
            "\n#{1,6} ",  # Markdown headers
            "```\n",      # Code blocks
            "\n\\*\\*\\*+\n",  # Asterisks separator
            "\n---+\n",   # Dash separator
            "\n___+\n",   # Underscore separator
            "\n\n",       # Paragraph break
            "\n",         # Line break
            " ",          # Space
            "",           # Default fallback
        ]
        
        # Configure text splitter
        text_splitters = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            encoding_name="cl100k_base",
            separators=MARKDOWN_SEPARATORS,
        )

        # Process each page in the document
        for page in document:
            if not hasattr(page, "page_content") or not hasattr(page, "metadata"):
                raise AttributeError("Each document object must have 'page_content' and 'metadata' attributes.")
            
            page_split = text_splitters.split_text(page.page_content)
            for page_sub_split in page_split:
                page_no = page.metadata.get('page', 0)  # Default to 0 if 'page' is missing
                metadata = {'source': "Uploaded_document", 'page_no': page_no + 1}
                doc_string = Document(page_content=page_sub_split, metadata=metadata)
                docs.append(doc_string)
        
        return docs

    except TypeError as e:
        print(f"TypeError: {e}")
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except Exception as e:
        print(f"Unexpected error in document_splitters: {e}")
    
    return None  # Return None in case of an error


def format_doc(docs):
    """
    Formats a list of Document objects into a single string with content joined by newlines.

    Args:
        docs (list): A list of Document objects with `page_content`.

    Returns:
        str: Formatted string containing all page contents joined by newlines.
    """
    if not docs:
        print("Warning: No documents to format. Returning an empty string.")
        return ""
    
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        print(f"Error formatting documents: {e}")
        return ""


# file_path = "data\documents\embeddings\My_Resume.pdf"
# document = load_document(file_path)
# print(document)