import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import google.generativeai as genai
import requests
from streamlit_lottie import st_lottie
from PIL import Image

# ========================= CONFIG =========================
st.set_page_config(
    page_title="GIG Assistant",
    page_icon="🧠",
    layout="centered"
)

# ========================= SIDEBAR =========================
page = st.sidebar.radio("Navigate", ["💬 Chat Assistant", "🖼️ Image Solver"])

# ========================= API SETUP =========================
def configure_gemini():
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
        st.stop()

configure_gemini()

# ========================= LOAD RAG =========================
@st.cache_resource
def load_rag():
    index = faiss.read_index("index.faiss")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    return index, chunks, model

index, chunks, embed_model = load_rag()

# ========================= HELPERS =========================
def load_lottie(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except:
        return None


def get_text_response(prompt: str):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {e}"


def get_vision_response(prompt: str, image):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {e}"


def retrieve_chunks(query, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# ========================= SYSTEM PROMPT =========================
SYSTEM_PROMPT = """
You are a helpful assistant supporting students of the GIG platform.

Hisham is the manager of the project of GIG.
Nile University is a research-based university in Egypt.

It offers programs in:
- Engineering & Applied Sciences
- Computer Science & AI
- Business & Digital Humanities
- Biotechnology

The university focuses on innovation and real-world impact.
Help students clearly and effectively.

Always, make the answer in the question language, if user asked a question in Arabic,
answer in Arabic, and if the user 
asked a question in English, answer in English.
"""

# ========================= PAGE 1: CHAT =========================
if page == "💬 Chat Assistant":

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Header
    st.markdown("<h3 style='text-align: center;'>Hello! I’m Gigo 🤖</h3>", unsafe_allow_html=True)

    # Animation
    lottie = load_lottie("https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json")
    if lottie:
        st_lottie(lottie, height=250)

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("What is your problem with the courses?")

    if user_input:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # RAG retrieval
        relevant_chunks = retrieve_chunks(user_input)
        context = "\n\n".join(relevant_chunks)

        # Prompt
        full_prompt = f"""
{SYSTEM_PROMPT}

You MUST answer using ONLY the following information:

{context}

If the answer is not in the data above, say:
"مش متأكد من الإجابة من البيانات المتاحة"

Question:
{user_input}
"""

        # Assistant response
        with st.chat_message("assistant"):
            reply = get_text_response(full_prompt)
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ========================= PAGE 2: IMAGE SOLVER =========================
elif page == "🖼️ Image Solver":

    st.title("🖼️ Screenshot Problem Solver")

    uploaded_file = st.file_uploader(
        "Upload a screenshot of your problem",
        type=["jpg", "jpeg", "png"]
    )

    VISION_PROMPT = """
You are an intelligent assistant.

Analyze the screenshot carefully and understand the problem.

Instructions:
- Explain the problem clearly
- Provide a simple step-by-step solution

Requirements:
- Answer in Arabic
- Use very simple words
- Avoid complex terms
- Be clear and concise
"""

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Screenshot")

        if st.button("Solve Problem"):
            with st.spinner("Analyzing..."):
                result = get_vision_response(VISION_PROMPT, image)

            st.success(result)






