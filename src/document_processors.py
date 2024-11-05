from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def text_splitter(pages):
    try:
        text_splitters = RecursiveCharacterTextSplitter(
            separators= "\n",
            chunk_size = 900,
            chunk_overlap = 100,
            length_function = len
        ) 
        splitted_text = text_splitters.split_documents(pages)
        return splitted_text
    except Exception as e:
        print(f"document cannot be splitted into chunks..{e}")
        return None
    
    

def document_loader(data):
    try:
        loader = PyPDFLoader(data)
        document = loader.load()
        return document
    except Exception as e:
        print(f"please upload the correct document...{e}")
        return None

