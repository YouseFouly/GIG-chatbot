import streamlit as st
import google.generativeai as genai
import requests
from streamlit_lottie import st_lottie

# -------------------- Configure Gemini API --------------------
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()

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

lottie_json = load_lottie_url(
    "https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json"
)

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="GIG",
    page_icon="🧠",
    layout="centered"
)

# -------------------- Initialize Chat History --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- Title --------------------
st.markdown("""
<div style='text-align: center;'>
    <h3>Hello! I’m Gigo, your virtual assistant</h3>
</div>
""", unsafe_allow_html=True)

# -------------------- Animation --------------------
if lottie_json:
    st_lottie(lottie_json, speed=1, loop=True, height=300)

# -------------------- Display Previous Messages --------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- Chat Input --------------------
user_prompt = st.chat_input("What is your problem with the courses?")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    # System Context
    system_prompt = """
You are a helpful assistant supporting students of the GIG platform.

Nile University is a research-based university in Egypt.
It offers undergraduate and graduate programs in:
my HR manager is nariman 
- Engineering & Applied Sciences
- Computer Science & Artificial Intelligence
- Business & Digital Humanities
- Biotechnology and other innovation-driven fields

The university focuses on research, innovation, entrepreneurship, and real-world impact.
Answer clearly and help students solve their academic problems.
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        response = model.generate_content(
            f"{system_prompt}\n\nStudent Question:\n{user_prompt}"
        )

        assistant_text = response.text.strip()

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_text}
        )

        with st.chat_message("assistant"):
            st.markdown(assistant_text)

    except Exception as e:
        st.error(f"❌ Error getting response: {e}")






