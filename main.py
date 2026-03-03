import os
import json
import streamlit as st
import google.generativeai as genai
import requests
from streamlit_lottie import st_lottie

# -------------------- Load Lottie Animation --------------------
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            st.error("❌ Failed to parse Lottie JSON.")
            return None
    else:
        st.error(f"❌ Lottie failed to load (status {response.status_code}).")
        return None

lottie_json = load_lottie_url("https://lottie.host/cc13fe2f-25c4-4547-b89a-5059f4044de4/Gn62lcFzBl.json")

# -------------------- Load API Key --------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir, "config.json")

try:
    with open(config_path) as f:
        config_data = json.load(f)
except FileNotFoundError:
    st.error("❌ config.json not found! Be sure it exists.")
    st.stop()

API_KEY = config_data.get("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in config.json")
    st.stop()

genai.configure(api_key=API_KEY)  # configure Gemini API

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GIG", page_icon="🧠", layout="centered")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
<div style='text-align: center;'>
    <h3>Hello! I’m Gigo — your Virtual Assistant</h3>
</div>
""", unsafe_allow_html=True)

if lottie_json:
    st_lottie(lottie_json, speed=1, loop=True, height=300)
else:
    st.error("❌ Failed to load animation.")

# show past messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("What is your problem with the courses?")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # system prompt + Nile University info
    system_prompt = """
You are a helpful assistant for students of the GIG platform.
Provide clear answers using Nile University info below when relevant:

Nile University is a national, research-based university in Egypt with schools in Engineering & Applied Sciences, Sciences, Digital Humanities, and others. It offers undergraduate and graduate programs focusing on innovation, research, and real-world impact. :contentReference[oaicite:0]{index=0}
"""

    try:
        # call Gemini
        response = genai.generate_content(
            model="gemini-2.5-flash-lite",
            prompt=f"{system_prompt}\nStudent question: {user_prompt}",
        )
        assistant_text = response.text.strip()

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

    except Exception as e:
        st.error(f"❌ Error getting response: {e}")