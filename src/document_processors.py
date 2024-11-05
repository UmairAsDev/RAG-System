from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitters = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap =30,
    # length_function = len,
    # is_separator_regex= False,
)


loader = PyPDFLoader("data\documents\embeddings\My Resume.pdf")
pages = loader.load()

split = text_splitters.split_documents(pages)

print(pages)

