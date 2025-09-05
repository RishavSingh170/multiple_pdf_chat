import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Remove any existing corrupted index
    faiss_index_path = "faiss_index"
    if os.path.exists(faiss_index_path):
        try:
            import shutil
            shutil.rmtree(faiss_index_path)
        except PermissionError:
            # If we can't delete, try to create a new one with a different name
            faiss_index_path = "faiss_index_new"
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(faiss_index_path)
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def generate_answer(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Check if FAISS index exists and is valid (check both possible paths)
        faiss_index_path = "faiss_index"
        if not os.path.exists(faiss_index_path) or not os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
            # Check for alternative index path
            faiss_index_path = "faiss_index_new"
            if not os.path.exists(faiss_index_path) or not os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
                return "No PDF files have been processed yet. Please upload and process PDF files first."
        
        try:
            new_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            return response["output_text"]
        except Exception as e:
            # If FAISS index is corrupted, try to delete it and ask user to reprocess
            import shutil
            try:
                if os.path.exists(faiss_index_path):
                    shutil.rmtree(faiss_index_path)
            except PermissionError:
                # If we can't delete due to permissions, just inform the user
                pass
            return f"Error loading the vector database. The index appears to be corrupted. Please upload and process your PDF files again. Error: {str(e)}"
    except Exception as e:
        return f"Error initializing the system. Please check your Google API key and try again. Error: {str(e)}"


def save_history_to_file(history, filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(history, f)


def load_history_from_file(filename="chat_history.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return [tuple(pair) for pair in data]
    return []


def download_chat_as_text(history):
    chat_text = ""
    for i, (q, a) in enumerate(history):
        chat_text += f"Q{i + 1}: {q}\nA{i + 1}: {a}\n\n"
    return chat_text
