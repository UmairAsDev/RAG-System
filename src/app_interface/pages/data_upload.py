import streamlit as st

def app():
    st.title("Data Upload")
    st.write("Upload your data files here.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'txt'])
    if uploaded_file:
        st.write("File uploaded successfully!")
        # Add code to process the uploaded file
