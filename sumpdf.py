import streamlit as st
import pdfplumber
import os
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

st.title("PDF Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("file.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("File uploaded successfully!")

    start_page = st.number_input("Enter the start page number:", min_value=1, step=1)
    end_page = st.number_input("Enter the end page number:", min_value=start_page, step=1)

    if st.button("Summarize"):
        model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
        tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

        filename = 'file.pdf'
        output_filename = "summary.txt"

        with pdfplumber.open(filename) as pdf:
            all_pages_text = ''
            for i, page in enumerate(pdf.pages):
                if i+1 >= start_page and i+1 <= end_page:
                    extracted_text = page.extract_text()
                    all_pages_text += extracted_text

        inputs = tokenizer([all_pages_text], truncation=True, return_tensors='pt')

        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=33, early_stopping=True, min_length=1400, max_length=8524)
        summarized_text = ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])

        # Save Summary
        with open(output_filename, 'w') as f:
            f.write(summarized_text[0])
        st.success(f"Summary of {filename} saved to {output_filename}")

        # Display Summary
        st.subheader("Summary:")
        st.write(summarized_text[0])
