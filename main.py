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

# ========================= API SETUP =========================
def configure_gemini():
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
        st.stop()

configure_gemini()

# ========================= HELPERS =========================
def load_lottie(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception:
        st.error("❌ Failed to load animation.")
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


# ========================= SESSION =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========================= UI =========================
st.markdown("""
<div style='text-align: center;'>
    <h3>Hello! I’m Gigo, your virtual assistant</h3>
</div>
""", unsafe_allow_html=True)

lottie = load_lottie(
    "https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json"
)

if lottie:
    st_lottie(lottie, height=200)

# ========================= CHAT =========================
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
"""

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("What is your problem with the courses?")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    full_prompt = f"{SYSTEM_PROMPT}\n\nStudent Question:\n{user_input}"
    reply = get_text_response(full_prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)

# ========================= SCREENSHOT SOLVER =========================
st.title("📸 Screenshot Problem Solver")

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

if uploaded_file and st.button("Solve Problem"):
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Screenshot")

    with st.spinner("Analyzing..."):
        result = get_vision_response(VISION_PROMPT, image)

    st.info(result)








