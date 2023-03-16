import streamlit as st
import pdfplumber
import os
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

st.title("PDF Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with open("file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded successfully!")

    start_page = st.number_input("Enter the start page number:", min_value=1, step=1)
    end_page = st.number_input("Enter the end page number:", min_value=start_page, step=1)
    output_filename = "sum_pagenrs.txt"

    if st.button("Extract and Summarize"):
        with pdfplumber.open("file.pdf") as pdf:
            all_pages_text = ""
            for i, page in enumerate(pdf.pages):
                if i + 1 >= start_page and i + 1 <= end_page:
                    extracted_text = page.extract_text()
                    all_pages_text += extracted_text

            with open(output_filename, "w") as f:
                f.write(all_pages_text)
            st.success("Text extraction complete.")

            model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
           


            try:
                import torch
            except ImportError:
                st.error("PyTorch library not found. Please install it and restart the runtime.")
            else:
                # The rest of the code that depends on PyTorch
                model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
                tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            
                inputs = tokenizer([all_pages_text], truncation=True, return_tensors="pt")
            
                summary_ids = model.generate(inputs["input_ids"], num_beams=10, early_stopping=True, min_length=600, max_length=10024)
                summarized_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]
            
                with open(output_filename, "w") as f:
                    f.write(summarized_text[0])
                st.success(f"Summary saved to {output_filename}")
                st.write(summarized_text[0])
