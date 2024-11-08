import streamlit as st
from pages import home, data_upload, data_analysis, query_interface, model_selection, history, about

st.set_page_config("RAG-PRO")

PAGES = {
    "Home": home,
    "Data Upload": data_upload,
    "Data Analysis": data_analysis,
    "Query Interface": query_interface,
    "Model Selection & Settings": model_selection,
    "History & Logs": history,
    "About": about
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Display the selected page
page = PAGES[selection]
page.app()
