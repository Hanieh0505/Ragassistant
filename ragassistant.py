import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖", layout="centered")

st.title("Book AI Assistant & RAG Chatbot")
st.write("A transparent, manual RAG system connected to your vector database to answer questions from your book.")


with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF and building vector store..."):
                
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                
                
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                
                
                st.cache_resource.clear()
                st.success("PDF processed successfully! You can now ask questions.")


@st.cache_resource
def rag_system():
    load_dotenv()
    api_key = os.getenv('API_GROQ_KEY')
    if not api_key:
        st.error("Please set your Groq API key in the .env file!")
        st.stop()
        
    groq_client = Groq(api_key=api_key)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    return groq_client, vector_store

groq_llm, anbari_store = rag_system()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_question := st.chat_input("Ask a question about the book..."):
    
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("🔍 Searching document and generating response..."):
        # Semantic search in Chroma vector store
        relevant_docs = anbari_store.similarity_search(user_question, k=4)

        
        context_parts = []
        sources = set()
        
        for doc in relevant_docs:
            context_parts.append(doc.page_content)
            page_num = doc.metadata.get("page", None)
            if page_num is not None:
                sources.add(str(page_num + 1)) # Pages usually start from 0

        context = "\n\n---\n\n".join(context_parts)

        
        conversation_history = ""
        for msg in st.session_state.messages[-5:]: # Last 5 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['content']}\n"

       
        prompt = f"""You are an intelligent AI assistant. Answer the user's question STRICTLY based on the provided document context below and the ongoing conversation history.
If the answer is not mentioned in the context, simply say: "The provided context does not contain enough information to answer this question."

Conversation History:
{conversation_history}

Document Context:
{context}

User Question: {user_question}
Answer:"""

        try:
            
            chat_completion = groq_llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="openai/gpt-oss-20b",
                temperature=1,
                max_completion_tokens=2048,
                reasoning_effort="medium",
            )
            
            response_text = chat_completion.choices[0].message.content

            
            if sources:
                sorted_sources = sorted(list(sources), key=int)
                response_text += f"\n\n*📄 **Relevant Sources (Pages):** {', '.join(sorted_sources)}*"

        except Exception as e:
            response_text = f"❌ An error occurred while communicating with the model: {e}"

    # Display assistant response and save to history
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})