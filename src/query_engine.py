chat_history = []

def chat_with_model(model, user_input, retrieved_docs):
    context = " ".join([doc.payload["doc_id"] for doc in retrieved_docs])
    input_text = context + "\nUser: " + user_input + "\nBot:"
    tokens = model['tokenizer'](input_text, return_tensors="pt")
    output = model['model'].generate(**tokens)
    response = model['tokenizer'].decode(output[0], skip_special_tokens=True)
    return response

def update_chat_history(user_message, bot_response):
    chat_history.append({"user": user_message, "bot": bot_response})
