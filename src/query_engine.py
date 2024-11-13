
from document_processors import load_document
from llm_embedings import embeddings
from vector_stores import vector_database
file_path = "data\documents\embeddings\My_Resume.pdf"
doc = load_document(file_path)

if doc is not None:
    doc_text = [document.page_content for document in doc]
    
    doc_text = " ".join(doc_text)
    
    embed_model = embeddings()
    if embed_model is not None:
        document_emdeddings = embed_model.encode(doc_text)
        
        database = vector_database(document_emdeddings)
        
        print(database)
    else:
        print("none")