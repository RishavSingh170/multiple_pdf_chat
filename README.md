# 📚 Chat with Multiple PDFs 

This project allows you to upload multiple PDF files and ask questions based on their content . It supports persistent chat history, context-based Q&A, and chat download functionality.
---

## 🚀 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/akshatt123/Multiple-PDF-Chat.git
   cd Multiple-PDF-Chat
   ```

2. **Create and activate a virtual environment (optional but recommended)**  
   ```bash
   conda create -n gemini_env python=3.10
   conda activate gemini_env
   ```

3. **Install the required packages**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Google API key**  
   - Create a `.env` file in the project root with the following content:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

---

## 🧠 How to Use

1. **Run the app**
   ```bash
   streamlit run app.py
   ```

2. **Features**
   - 📂 Upload multiple PDF files from the sidebar
   - ❓ Ask questions about the uploaded documents
   - 💬 View chat history with expandable Q&A
   - 🔄 Clear chat history
   - 📥 Download chat as a `.txt` file with a custom filename

---

## ✅ Requirements

- Python > 3.10
- Streamlit
- LangChain
- PyPDF2
- FAISS
- dotenv
- google-generativeai

---

## 🔮 Future Work

- Add support for audio-based question input
- Allow user login and individual chat history storage
- Add charts/visuals based on document content
