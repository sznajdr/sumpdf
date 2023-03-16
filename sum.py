import streamlit as st
import pdfplumber
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import os

# Create output directory if it does not exist
if not os.path.exists("sums"):
    os.makedirs("sums")

# Define summarization function
def summarize_text(text, length):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, length)
    return summary

# Page selection form
def page_selection_form():
    with st.form("page_selection"):
        st.header("Page Selection")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        start_page = st.number_input("Enter the start page number", value=1, min_value=1)
        end_page = st.number_input("Enter the end page number", value=1, min_value=1)
        submit_button = st.form_submit_button(label="Extract Text")
    return uploaded_file, start_page, end_page, submit_button

# Summarization form
def summarization_form():
    with st.form("summarization"):
        st.header("Summarization")
        uploaded_text = st.session_state.uploaded_text
        length = st.number_input("Enter the length of summary", value=3, min_value=1)
        submit_button = st.form_submit_button(label="Summarize Text")
    return uploaded_text, length, submit_button

# Main app
def app():
    st.title("PDF Summarizer")
    uploaded_file, start_page, end_page, submit_button = page_selection_form()
    
    # Display text when user submits page selection form
    if submit_button and uploaded_file is not None:
        with st.spinner("Extracting text..."):
            with pdfplumber.open(uploaded_file) as pdf:
                all_pages_text = ''
                for i, page in enumerate(pdf.pages):
                    if i+1 >= start_page and i+1 <= end_page:
                        extracted_text = page.extract_text()
                        all_pages_text += extracted_text
                st.session_state.uploaded_text = all_pages_text
        st.success("Text extracted successfully.")
    
    # Display summarization form when text is uploaded
    if "uploaded_text" in st.session_state:
        uploaded_text, length, submit_button = summarization_form()
        
        # Display summary when user submits summarization form
        if submit_button and uploaded_text is not None:
            with st.spinner("Summarizing text..."):
                summary = summarize_text(uploaded_text, length)
                summary_text = '\n'.join([str(s) for s in summary])
            st.success("Text summarized successfully.")
            st.write(summary_text)
            
# Run app
if __name__ == "__main__":
    app()
