# 📚 Manual RAG Book AI Assistant (Streamlit & Groq)

A transparent, manual Retrieval-Augmented Generation (RAG) system built with Python, LangChain, ChromaDB, and Groq API, featuring an interactive Streamlit UI, conversation history, and source page citations.

## 🚀 Features
- **Manual RAG Pipeline:** No black-box abstractions; clean and transparent implementation of document loading, text splitting, embeddings, and vector search.
- **Interactive Web UI:** Built with Streamlit for a smooth chat experience.
- **Dynamic PDF Upload:** Upload any PDF document directly from the sidebar to instantly build a vector database.
- **Chat History:** Multi-turn conversational memory using Streamlit session state.
- **Source Citations:** Automatically displays the source page numbers for each answer.

## 🛠️ Tech Stack
- **Python**
- **Streamlit**
- **LangChain** (Community, Text Splitters, HuggingFace Embeddings, Chroma)
- **Groq API** (Integration with high-performance/reasoning models)

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME

Install dependencies:

B
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma groq python-dotenv streamlit
Configure Environment Variables:
Create a .env file in the root directory and add your Groq API key:


API_GROQ_KEY=your_actual_groq_api_key_here

run with:
streamlit run app.py