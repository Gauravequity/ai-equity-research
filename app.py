import streamlit as st
from PyPDF2 import PdfReader
import openai
import whisper
import requests
import os
import chromadb
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ratelimit import limits, sleep_and_retry

# 🔹 Load API Keys Securely from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# 🔹 Initialize OpenAI Client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔹 Initialize ChromaDB for Document Search
chroma_client = chromadb.PersistentClient(path="chroma_db")
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
collection = chroma_client.get_or_create_collection("research_docs")

# 🔹 Streamlit UI Enhancements
st.set_page_config(page_title="AI Equity Research", layout="wide")
st.title("📊 AI-Powered Equity Research Chat")

# 🔹 Guide & Example Queries
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### 🛠️ How to Use:
    1️⃣ **Upload Documents** – Add PDFs (e.g., annual reports) or audio (e.g., earnings calls).  
    2️⃣ **Ask Questions** – Query the uploaded content for insights.  
    3️⃣ **AI Analysis** – The tool searches & provides key insights.  
    4️⃣ **Chat History** – Left panel saves past queries.  
    5️⃣ **Web Search (If Needed)** – Fetches extra insights from Google.  
    """)
with col2:
    st.markdown("""
    🔹 **Example Queries:**  
    - *"Summarize this annual report."*  
    - *"What risks does the management highlight?"*  
    - *"What are the revenue trends?"*
    """)

# 🔹 Sidebar: Chat History
st.sidebar.header("💬 Chat History")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    st.sidebar.markdown(f"**User:** {chat['user']}")
    st.sidebar.markdown(f"**AI:** {chat['ai']}")
    st.sidebar.markdown("---")

# 🔹 File Upload UI
st.markdown("### 📂 Upload Your Financial Reports & Earnings Calls")
uploaded_files = st.file_uploader(
    "Upload PDFs and/or Audio Files (MP3/WAV)", type=["pdf", "mp3", "wav"], accept_multiple_files=True
)

# 🔹 Process Uploaded Files
if uploaded_files and "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = True  # Prevent re-processing

    with st.spinner("🔄 Processing uploaded files... Please wait."):
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type

            if file_type == "application/pdf":
                st.markdown(f"📄 **{uploaded_file.name}**")
                pdf_reader = PdfReader(uploaded_file)
                pdf_text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])

                # 🔹 Use OpenAI Embeddings for ChromaDB
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(pdf_text)

                # Batch Embedding for Faster Processing
                embeddings = embedding_function.embed_documents(chunks)
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    collection.add(
                        ids=[f"{uploaded_file.name}_chunk_{i}"],
                        metadatas=[{"filename": uploaded_file.name}],
                        documents=[chunk],
                        embeddings=[embedding]
                    )

            elif file_type in ["audio/mpeg", "audio/wav"]:
                st.markdown(f"🎙️ **{uploaded_file.name}**")
                file_path = f"temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 🔹 Transcribe Audio
                st.success(f"Processing {uploaded_file.name}...")
                model = whisper.load_model("base")
                result = model.transcribe(file_path)
                transcript_text = result["text"]

                # 🔹 Store Transcript in ChromaDB
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(transcript_text)

                embeddings = embedding_function.embed_documents(chunks)
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    collection.add(
                        ids=[f"{uploaded_file.name}_chunk_{i}"],
                        metadatas=[{"filename": uploaded_file.name}],
                        documents=[chunk],
                        embeddings=[embedding]
                    )

                os.remove(file_path)  # Cleanup

    st.success("✅ Files stored for analysis!")

# 🔹 Chat Input UI
st.markdown("## 💬 What do we research today?")
user_input = st.text_input(
    "🧐 Start with adding documents or shoot away with your queries!", key="chat_input"
)

# 🔹 AI Model Selection
st.subheader("🤖 Choose AI Model")
model_choice = st.radio("Select AI Model:", ["DeepSeek (Free)", "ChatGPT (Paid)"], horizontal=True)

# 🔹 Stream AI Responses Function
@sleep_and_retry
@limits(calls=20, period=60)  # Prevent excessive API calls
def stream_ai_response(prompt, model):
    """ Streams AI responses """
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True
    )
    return response

# 🔹 Process User Query with ChromaDB
if user_input:
    with st.spinner("🔍 Searching documents..."):
        search_results = collection.query(query_texts=[user_input], n_results=10)
        matched_text = "\n\n".join(search_results["documents"][0]) if search_results["documents"] else "No relevant content found."

    # 🔹 AI Response
    st.subheader("🗨️ Chat Response")
    if model_choice == "ChatGPT (Paid)":
        response = stream_ai_response(f"Answer based on these excerpts:\n\n{matched_text}", "gpt-4.5-preview")
        answer = "".join([chunk["choices"][0]["delta"].get("content", "") for chunk in response])
    else:
        deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": user_input}], "max_tokens": 300}
        response = requests.post(deepseek_url, json=payload, headers=headers)
        answer = response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else "DeepSeek API Error."

    st.write(answer)

    # 🔹 Save Chat History
    st.session_state.chat_history.append({"user": user_input, "ai": answer})
    st.experimental_rerun()
