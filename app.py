import streamlit as st
from model_utils import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    generate_answer,
    save_history_to_file,
    load_history_from_file,
    download_chat_as_text
)

st.set_page_config(page_title="Chat with Multiple PDF",
                   page_icon='🤖')

def main():
    st.header("Chat with Multiple PDFs ")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_history_from_file()
    
    if "pdfs_processed" not in st.session_state:
        st.session_state.pdfs_processed = False

    st.subheader("Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {q}"):
            st.markdown(f"**A{i+1}:** {a}")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Ask") and user_question:
        if not st.session_state.pdfs_processed:
            st.error("Please upload and process PDF files first before asking questions.")
        else:
            answer = generate_answer(user_question)
            st.session_state.chat_history.append((user_question, answer))
            save_history_to_file(st.session_state.chat_history)
            st.write("**You:**", user_question)
            st.write("**Assistant:**", answer)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file before processing.")
            else:
                try:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No text could be extracted from the uploaded PDFs. Please check if the PDFs are valid and not password-protected.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.session_state.pdfs_processed = True
                            st.success("PDFs processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            save_history_to_file([])
            st.success("Chat history cleared.")
            st.rerun()
        
        if st.button("Reset PDF Processing"):
            st.session_state.pdfs_processed = False
            # Try to remove any existing index
            import shutil
            import os
            for index_path in ["faiss_index", "faiss_index_new"]:
                if os.path.exists(index_path):
                    try:
                        shutil.rmtree(index_path)
                    except:
                        pass
            st.success("PDF processing status reset. Please upload and process PDFs again.")
            st.rerun()
        
        # Status indicator
        if st.session_state.pdfs_processed:
            st.success("✅ PDFs are processed and ready for questions!")
        else:
            st.warning("⚠️ No PDFs processed yet. Upload and process PDFs to start asking questions.")

        if "filename_input" not in st.session_state:
            st.session_state.filename_input = "my_chat.txt"

        st.session_state.filename_input = st.text_input(
            "Enter filename for download",
            value=st.session_state.filename_input,
            key="filename_text_input"
        )

        if st.session_state.chat_history:
            chat_text = download_chat_as_text(st.session_state.chat_history)
            file_name = st.session_state.filename_input.strip()
            if not file_name.endswith(".txt"):
                file_name += ".txt"

            st.download_button(
                label="📄 Download Chat History",
                data=chat_text,
                file_name=file_name,
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
