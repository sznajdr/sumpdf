import streamlit as st
import pdfplumber
import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def summarize_file(text, length):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, length)
    return summary

st.title("PDF Summarizer")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        with pdfplumber.open(uploaded_file) as pdf:
            all_pages_text = ''
            for i, page in enumerate(pdf.pages):
                extracted_text = page.extract_text()
                all_pages_text += extracted_text
    st.success("Text extraction complete.")

    start_page = st.number_input("Enter the start page number:", min_value=1, value=1)
    end_page = st.number_input("Enter the end page number:", min_value=start_page, value=start_page)
    summary_length = st.number_input("Enter the summary length:", min_value=1, value=10)

    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summary = summarize_file(all_pages_text, summary_length)
            st.write("Summary:")
            for sentence in summary:
                st.write(sentence)
        st.success("Summary complete.")
