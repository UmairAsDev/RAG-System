from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


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
        else:
            print("Unsupported document format.")
            return None

        return content 

    except Exception as e:
        print(f"Cannot load document... {e}")
        return None

    
   
   
 
def document_splitters(document):
    try:
        docs = []
        text_splitters = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = 1000,
            chunk_overlap = 100,
            is_separator_regex= False,
            encoding_name="cl100k_base",
        )
        
        for page in document:
            page_split = text_splitters.split_text(page.page_content)
            for page_sub_split in page_split:
                page_no = page.metadata.get('page', 0)
                metadata = {'source': "Uploaded_document", 'page_no' : page_no + 1}
                doc_string = Document(page_content=page_sub_split, metadata=metadata)
                docs.append(doc_string)
        return docs
    except Exception as e:
        print(f"cannot split the document..{e}")
        return None
    


    