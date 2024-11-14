from document_processors import load_document, document_splitters, format_doc
from vector_stores import vector_database

# from text_retrievar import retrieval_engine
import streamlit as st

st.set_page_config("Conversational AI")
st.title("Multi-Rag-System")


def main():

    try:
        upload_file = st.sidebar.file_uploader(
            "Upload Your Document", type=["pdf", "txt", "docx", "csv"]
        )
        if upload_file:
            file_path = f"{upload_file.name}"
            with open(file_path, "wb") as f:
                f.write(upload_file)

            document = load_document(file_path)
            if document:
                split_docs = document_splitters(document)

                if split_docs:
                    doc_list = format_doc(split_docs)
                    store_document = vector_database(doc_list)
                    print(f"document stored in the databse..{store_document}")

                # model_name = st.sidebar.selectbox(
                #     "Select LLM",
                #     [
                #         "Reformer",
                #         "Blenderbot",
                #         "T5",
                #         "Bart",
                #         "DistilGPT-2",
                #         "DialoGPT",
                #         "GPT2",
                #     ],
                # )
                
    except Exception as e:
        print(f"document can't be loaded{e}")


if __name__ == "__main__":
    main()
