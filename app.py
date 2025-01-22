import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # Store PDF as tempfile
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    try:
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Ensure the file is properly closed before processing

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        return text_splitter.split_documents(docs)
    finally:
        # Delete the temporary file after processing
        os.unlink(temp_file.name)


if __name__ == "__main__":
    st.set_page_config(page_title="Local RAG Application")
    st.sidebar.header("Local RAG QA System")
    
    uploaded_file = st.sidebar.file_uploader(
        "**Upload PDF file for QnA**", type=["pdf"], accept_multiple_files=False
    )
    
    process = st.sidebar.button("Process")
    
    if uploaded_file and process:
        try:
            all_splits = process_document(uploaded_file)
            st.write(all_splits)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
