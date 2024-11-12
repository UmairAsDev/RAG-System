from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader


def load_document(document):
    try:
        if document.endswith(".pdf"):
            loader = PyPDFLoader(document)
            content = loader.load()
        elif document.endswith(".docx"):
            loader = Docx2txtLoader(document)
            content = loader.load()
        elif document.endswith(".csv"):
            loader = CSVLoader(document)
            content = loader.load()
        elif document.endswith(".txt"):
            loader = TextLoader(document)
            content = loader.load()   
            return content
    except Exception as e:
        print(f"Cannot load Document...{e}")
        return None
    
   
   
from langchain_text_splitters import RecursiveCharacterTextSplitter 
def document_splitters(document):
    try:
        text_splitters = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = 1000,
            chunk_overlap = 100,
            is_separator_regex= False,
            encoding_name="cl100k_base",
        )
        content = text_splitters.split_text(document)
        return content
    except Exception as e:
        print(f"cannot split the document..{e}")
        return None
    

document = "this is the new text for splitting\n?"

splitted = document_splitters(document)
print(splitted)
