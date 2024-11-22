import tempfile
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
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


from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document


def split_document(document):
    try:
        docs = []
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
        )

        page_split = text_splitter.split_text(document.page_content)
        for page_sub_split in page_split:
            metadata = {'source': document.metadata.get('source', 'unknown')}
            doc_string = Document(page_content=page_sub_split, metadata=metadata)
            docs.append(doc_string)

        return docs
    except Exception as e:
        print(f"Failed to split the document: {e}")
        return None
    


from llm_embedings import embeddings
from langchain_qdrant import QdrantVectorStore
import yaml

def store_in_vector_db(docs):
    try:
        with open(r'../config/qdrant.yaml', 'r') as file:
            secrets = yaml.safe_load(file)
            url = secrets["QDRANT_URL"]
            api_key = secrets["QDRANT_API"]
            
        all_docs = []
        for doc in docs:
            split_docs = split_document(doc)
            all_docs.extend(split_docs)

        embedding = embeddings()
        qdrant_client = QdrantVectorStore.from_documents(
            all_docs,
            embedding,
            url=url,
            prefer_grpc=True,
            api_key=api_key,
            collection_name="Rag_system",
        )
                
        return qdrant_client
    except yaml.YAMLError as yaml_error:
        print(f"Error Loading Configuration: {yaml_error}")
        return None
    except Exception as e:
        print(f"Vector store is not created..{e}")
        return None



def cached_retrieval(query, qdrant_client):
    try:
        # Perform a retrieval from the vector database using the query
        # For simplicity, assuming qdrant_client has a method 'retrieve' that retrieves the documents
        retrieved_docs = qdrant_client.similarity_search(query, k=5)  # Adjust `k` as needed
        return retrieved_docs
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        return []

def truncate_text(text, max_tokens=500):
    words = text.split()
    return " ".join(words[:max_tokens]) if len(words) > max_tokens else text

def get_user_input(is_streamlit):
    return st.text_input("Your query:") if is_streamlit else input("User: ")




def display_response(response, is_streamlit):
    if is_streamlit:
        st.write("AI:", response)
    else:
        print("AI:", response)


from transformers import pipeline
summarizer = pipeline("summarization")
def summarize_text(text):
    summarized_text = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summarized_text[0]['summary_text']




from transformers import pipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFacePipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def invoke(self, prompt_input):
        result = self.pipeline(prompt_input, max_length=200, truncation=True)
        return result[0]['generated_text']

def conversation(model_name, history, qdrant_client):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt_template = PromptTemplate(
        input_variables=["history", "query", "retrieved_docs"],
        template=(
            "Conversation history:\n{history}\n\n"
            "Relevant documents:\n{retrieved_docs}\n\n"
            "User: {query}\nAI:"
        ),
    )

    def handle_query(user_query):
        try:
            greetings = ["hi", "hello", "hey", "how are you"]
            if any(greeting in user_query.lower() for greeting in greetings):
                ai_response = "Hi there! How can I assist you today?"
                return {"response": ai_response, "retrieved_docs": ""}

            retrieved_docs = cached_retrieval(user_query, qdrant_client)
            print(f"Retrieved {len(retrieved_docs)} documents")

            if not retrieved_docs:
                print("No documents retrieved.")
                retrieved_text = "No relevant documents found."
            else:
                retrieved_text = "\n".join([summarize_text(doc.page_content) for doc in retrieved_docs])

            prompt_input = prompt_template.format(history=history, query=user_query, retrieved_docs=retrieved_text)
            ai_response = pipe.invoke(prompt_input)

            print(f"AI Response: {ai_response}")

            return {"response": ai_response, "retrieved_docs": retrieved_text}
        except Exception as e:
            print(f"Error in handle_query: {e}")
            return {"response": "An error occurred while processing your query.", "retrieved_docs": ""}
    return handle_query



import streamlit as st

def main():
    st.set_page_config(page_title="Conversational AI", page_icon="🤖", layout="wide")
    st.header("Conversational AI")

    try:
        uploaded_files = st.sidebar.file_uploader(
            "Upload Your Document",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True
        )

        if "history" not in st.session_state:
            st.session_state.history = ""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "qdrant_client" not in st.session_state:
            st.session_state.qdrant_client = None

        if uploaded_files:
            document = load_document(uploaded_files)
            if document:
                split_docs = split_document(document)
                if split_docs:
                    qdrant_client = store_in_vector_db(split_docs)
                    if qdrant_client:
                        st.session_state.qdrant_client = qdrant_client
                        st.success("Document successfully loaded and stored in the database.")
                    else:
                        st.error("Failed to initialize the vector database.")
                else:
                    st.error("Failed to split the document.")
            else:
                st.error("Failed to load document.")

        model_name = st.sidebar.selectbox(
            "Select LLM",
            ["gpt2", "distilgpt2", "openai-gpt", "gpt-neo-125M"],
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_query = st.chat_input("Type your message here...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            if st.session_state.qdrant_client:
                conversation_handler = conversation(
                    model_name, st.session_state.history, st.session_state.qdrant_client
                )
                result = conversation_handler(user_query)
                ai_response = result["response"]

                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.write(ai_response)

                st.session_state.history += f"\nUser: {user_query}\nAI: {ai_response}"
            else:
                st.warning("Please upload and process documents before starting the conversation.")

    except Exception as e:
        st.error(f"An error")
        
if __name__ == "__main__":
    main()