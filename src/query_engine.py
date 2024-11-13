
from document_processors import load_document, document_splitters
from llm_embedings import embeddings
from vector_stores import vector_database

file_path = "data/documents/embeddings/My_Resume.pdf"
doc = load_document(file_path)

# if doc is not None:
#     doc_text = [document.page_content for document in doc]
    
#     doc_text = " ".join(doc_text)
    
#     embed_model = embeddings()
#     if embed_model is not None:
#         document_emdeddings = embed_model.embed_query(doc_text)
        
#         database = vector_database(document_emdeddings)
        
#         print(database)
#     else:
#         print("none")
        
print(f"document loaded sucessfully {doc}")
if doc is not None:
    splitted = document_splitters(doc)
    print(f"the process is sucessful...{splitted}")
    if splitted is not None:
        data = vector_database(splitted)
    else:
        print("there must be some issue...")
        
    