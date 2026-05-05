import streamlit as st
from streamlit_option_menu import option_menu
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# ========================= CONFIG =========================
st.set_page_config(page_title="GIG Assistant", page_icon="🧠", layout="centered")

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 5                    # retrieve more chunks for better coverage
SIMILARITY_THRESHOLD = 0.35  # discard chunks below this cosine score (0–1)

# ========================= API =========================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ API Key not found. Add GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()


# ========================= LOAD RAG =========================
@st.cache_resource
def load_rag():
    try:
        index = faiss.read_index("index.faiss")

        with open("chunks.pkl", "rb") as f:
            entries = pickle.load(f)   # list of dicts: {text, lang, source}

        model = SentenceTransformer(EMBED_MODEL_NAME)
        return index, entries, model

    except Exception as e:
        st.error(f"❌ Error loading RAG: {e}")
        st.stop()

index, entries, embed_model = load_rag()


# ========================= RETRIEVAL =========================
def retrieve_chunks(query: str, k: int = TOP_K) -> list[str]:
    """
    Retrieves top-k most relevant chunks using cosine similarity.
    - Uses multilingual-e5 'query: ' prefix for correct retrieval behavior.
    - Filters out chunks below SIMILARITY_THRESHOLD to avoid noise.
    - Deduplicates near-identical chunks.
    """
    try:
        query_embedding = embed_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(query_embedding, k)

        results = []
        seen = set()

        for score, idx in zip(scores[0], indices[0]):
            if score < SIMILARITY_THRESHOLD:
                continue                        # skip irrelevant chunks

            text = entries[idx]["text"]

            # Simple deduplication: skip if text is >80% similar to an existing result
            normalized = " ".join(text.split()[:30])  # fingerprint: first 30 words
            if normalized in seen:
                continue
            seen.add(normalized)

            results.append(text)

        return results

    except Exception:
        return []


# ========================= LLM CALLS =========================
def get_text_response(system_prompt: str, conversation_history: list[dict]) -> str:
    """
    Sends full conversation history to Gemini so the model has memory context.
    conversation_history format: [{"role": "user"|"model", "parts": ["text"]}]
    """
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash-lite",
            system_instruction=system_prompt
        )
        chat = model.start_chat(history=conversation_history[:-1])
        last_message = conversation_history[-1]["parts"][0]
        response = chat.send_message(last_message)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"


def get_vision_response(prompt: str, image) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"❌ Error analyzing image: {e}"


def load_lottie(url: str):
    try:
        return requests.get(url).json()
    except Exception:
        return None


# ========================= SYSTEM PROMPT =========================
SYSTEM_PROMPT = """
You are GIGO, a helpful academic assistant for the GIG platform at Nile University.

Instructions:
1. Use the provided CONTEXT as your PRIMARY source of truth.
2. Reason carefully over the context before answering.
3. If the answer is clearly present in the context, answer directly and concisely.
4. If the answer is NOT clearly found in the context, say:
   - In Arabic: "لم أجد إجابة واضحة في البيانات المتاحة، لكن يمكنني مساعدتك بشكل عام."
   - In English: "I couldn't find a clear answer in the available data, but I can help generally."
5. ALWAYS answer in the SAME language the user used in their question.
6. Be clear, concise, and academic in tone.
7. Do NOT fabricate information that is not in the context.
"""


# ========================= SIDEBAR =========================
with st.sidebar:
    selected = option_menu(
        "GIG Assistant",
        ["Chat Assistant", "Image Solver"],
        menu_icon="cpu",
        icons=["chat-dots-fill", "image-fill"],
        default_index=0
    )


# ========================= PAGE 1: CHAT =========================
if selected == "Chat Assistant":

    st.title("🤖 GIGO")
    st.markdown("### Hello! I'm your friend Gigo, how can I help you?")

    lottie = load_lottie("https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json")
    if lottie:
        st_lottie(lottie, height=220)

    # Initialize chat history in session state
    # Format: list of {"role": "user"|"assistant", "text": str, "gemini_role": "user"|"model"}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    user_input = st.chat_input("Ask your question here... / اكتب سؤالك هنا...")

    if user_input:
        # Show user message immediately
        st.session_state.chat_history.append({
            "role": "user",
            "text": user_input,
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant context
        relevant_chunks = retrieve_chunks(user_input)

        if relevant_chunks:
            context = "\n\n---\n\n".join(relevant_chunks)
        else:
            context = "No relevant content found in the knowledge base."

        # Build the current turn's full content (context + question)
        augmented_question = f"""CONTEXT:
{context}

QUESTION:
{user_input}"""

        # Build Gemini conversation history (all prior turns + current augmented question)
        # Gemini uses "user" / "model" roles
        gemini_history = []
        for msg in st.session_state.chat_history[:-1]:   # exclude current turn
            gemini_role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": gemini_role,
                "parts": [msg["text"]]
            })

        # Add the current augmented turn
        gemini_history.append({
            "role": "user",
            "parts": [augmented_question]
        })

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_text_response(SYSTEM_PROMPT, gemini_history)
                st.markdown(reply)

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": reply,
        })

    # Optional: clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ========================= PAGE 2: IMAGE SOLVER =========================
elif selected == "Image Solver":

    st.title("🖼️ Image Problem Solver")
    st.markdown("Upload a screenshot and get a simple explanation")

    lottie = load_lottie("https://lottie.host/0db5a9c4-6c54-4c65-bb3d-66d7dc5a4d0e/0Tzv7FpC18.json")
    if lottie:
        st_lottie(lottie, height=220)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    VISION_PROMPT = """
Analyze the image carefully and explain the problem shown.

Requirements:
- Answer in Arabic
- Provide a simple, clear explanation of what the image shows
- Give a step-by-step solution if applicable
- Use academic language appropriate for a university student
"""

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Solve Problem"):
            with st.spinner("Analyzing image..."):
                result = get_vision_response(VISION_PROMPT, image)
            st.success(result)


