import streamlit as st
from streamlit_option_menu import option_menu
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# ========================= CONFIG =========================
st.set_page_config(page_title="GIG Assistant", page_icon="🧠", layout="centered")

# ========================= API =========================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("❌ API Key not found. Add GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()

# ========================= LOAD RAG =========================
@st.cache_resource
def load_rag():
    try:
        index = faiss.read_index("index.faiss")

        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        # ✅ Multilingual model (fix Arabic issue)
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        return index, chunks, model
    except Exception as e:
        st.error(f"❌ Error loading RAG: {e}")
        st.stop()

index, chunks, embed_model = load_rag()

# ========================= HELPERS =========================
def retrieve_chunks(query, k=3):
    try:
        query_embedding = embed_model.encode([query])
        distances, indices = index.search(query_embedding, k)
        return [chunks[i] for i in indices[0]]
    except:
        return []


def get_text_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"


def get_vision_response(prompt, image):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"❌ Error analyzing image: {e}"


def load_lottie(url):
    try:
        return requests.get(url).json()
    except:
        return None


# ========================= SIDEBAR =========================
with st.sidebar:
    selected = option_menu(
        "GIG Assistant",
        ["Chat Assistant", "Image Solver"],
        menu_icon="cpu",
        icons=["chat-dots-fill", "image-fill"],
        default_index=0
    )

# ========================= SYSTEM PROMPT =========================
SYSTEM_PROMPT = """
You are a helpful assistant for the GIG platform (Nile University).

Rules:
- Use the provided context as your MAIN source.
- If the answer is not clearly found, say:
    Arabic: "مش متأكد من الإجابة من البيانات المتاحة"
    English: "I'm not sure based on the available data"
- Answer in the SAME language as the user.
- Be clear, simple, and helpful.
"""

# ========================= PAGE 1: CHAT =========================
if selected == "Chat Assistant":

    st.title("🤖 GIG Chat Assistant")
    st.markdown("### Hello! I’m Gigo 🤖 — Ask me anything about the GIG program")

    # Animation
    lottie = load_lottie("https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json")
    if lottie:
        st_lottie(lottie, height=220)

    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # Input
    user_input = st.chat_input("Ask your question here...")

    if user_input:
        # Show user message
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve context
        relevant_chunks = retrieve_chunks(user_input)

        if not relevant_chunks:
            context = "No relevant data found."
        else:
            context = "\n\n".join(relevant_chunks)

        # Build prompt
        full_prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{user_input}
"""

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_text_response(full_prompt)
                st.markdown(reply)

        st.session_state.chat_history.append(("assistant", reply))


# ========================= PAGE 2: IMAGE SOLVER =========================
elif selected == "Image Solver":

    st.title("🖼️ Image Problem Solver")
    st.markdown("Upload a screenshot and get a simple explanation")

    # Animation
    lottie = load_lottie("https://lottie.host/0db5a9c4-6c54-4c65-bb3d-66d7dc5a4d0e/0Tzv7FpC18.json")
    if lottie:
        st_lottie(lottie, height=220)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    VISION_PROMPT = """
Analyze the image and explain the problem.

Requirements:
- Answer in Arabic
- Simple explanation
- Step-by-step solution
"""

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Solve Problem"):
            with st.spinner("Analyzing image..."):
                result = get_vision_response(VISION_PROMPT, image)

            st.success(result)




