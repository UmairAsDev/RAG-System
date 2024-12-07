from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile
import streamlit as st

def load_document(uploaded_files):
    try:
        all_contents = []
        for uploaded_file in uploaded_files:
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
                st.error(f"Unsupported document format: {uploaded_file.name}")
                continue

            all_contents.extend(content)

        return all_contents if all_contents else None

    except Exception as e:
        st.error(f"Cannot load document... {e}")
        return None



def document_splitters(document):
    try:
        docs = []
        MARKDOWN_SEPARATORS = [
            "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]
        text_splitters = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            encoding_name="cl100k_base",
            separators=MARKDOWN_SEPARATORS,
        )

        for page in document:
            page_split = text_splitters.split_text(page.page_content)
            for page_sub_split in page_split:
                page_no = page.metadata.get('page', 0)
                metadata = {'source': document, 'page_no': page_no + 1}
                doc_string = Document(page_content=page_sub_split, metadata=metadata)
                docs.append(doc_string)
        return docs
    except Exception as e:
        print(f"cannot split the document..{e}")
        return None




from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document

def alternative_text_splitter(document, chunk_size=1000, chunk_overlap=100):
    try:
        docs = []
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
        )

        page_split = text_splitter.split_text(document.page_content)
        for page_sub_split in page_split:
            page_no = document.metadata.get('page', 0)
            metadata = {'source': document, 'page_no': page_no + 1}
            doc_string = Document(page_content=page_sub_split, metadata=metadata)
            docs.append(doc_string)

        return docs
    except Exception as e:
        print(f"Cannot split the document: {e}")
        return None


from transformers import pipeline

summarizer = pipeline("summarization")

def summarize_text(text):
    summarized_text = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summarized_text[0]['summary_text']


