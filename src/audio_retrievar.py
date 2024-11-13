import speech_recognition as sr
from gtts import gTTS
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def audio_query_response():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    # Capture audio input
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)  # Automatically adjusts for ambient noise
        print("Listening...")
        audio = recognizer.listen(source, timeout=100000)  # Set a higher timeout (default is 5 seconds)
        query = recognizer.recognize_sphinx(audio)
        print(f"Query: {query}")


    # Generate response using the model
    inputs = tokenizer.encode(query, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=50)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    print("Response:", response)

    # Convert response to audio and play
    tts = gTTS(response)
    tts.save("response.mp3")
    os.system("start response.mp3")  # Or use a different play command if needed

audio_query_response()
