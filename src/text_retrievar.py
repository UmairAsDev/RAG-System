from langchain.chains.llm import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
from llm_manager import load_model
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from transformers import pipeline
import torch
from vector_stores import vector_database
from document_processors import load_document, document_splitters, format_doc

def retrieval_engine(model_name, docs):
    # Load language model
    model, tokenizer = load_model(model_name)
    if model is None or tokenizer is None:
        print("Failed to load model. Please check the model name and try again.")
        return

    # Set up text generation pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=100,
        max_length=200
    )
    pipe = HuggingFacePipeline(pipeline=hf_pipeline)

    # Set up prompt template
    prompt_template = PromptTemplate(
        input_variables=["history", "query", "retrieved_docs"],
        template="conversation history:\n{history}\n\nRelevant documents:\n{retrieved_docs}\n\nUser: {query}\nAI:"
    )

    memory = ConversationBufferWindowMemory(memory_key="history", k=5)
    conversation_chain = LLMChain(
        llm=pipe,
        prompt=prompt_template,
        memory=memory
    )

    # Initialize vector database
    qdrant_client = vector_database(docs)
    if qdrant_client is None:
        print("Vector database not initialized. Exiting.")
        return

    print("Start chatting with the AI! Type 'exit' or 'quit' to end the conversation.")
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "bye", "stop"]:
            print("Ending the conversation. Goodbye!")
            break
        
        # Retrieve relevant documents
        retrieved_docs = qdrant_client.similarity_search(query, top_k=3)
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

        # Run conversation chain with retrieved documents as context
        response = conversation_chain.run(query=query, retrieved_docs=retrieved_text)
        print("AI:", response)

# Example usage
# file_path = "data\documents\embeddings\My_Resume.pdf"
# loader = load_document(file_path)
# splitter = document_splitters(loader)
# formatted = format_doc(splitter)
# retrieval_engine("GPT2", formatted)
