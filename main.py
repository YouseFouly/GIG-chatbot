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

st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}

/* Fix chat message layout */
.stChatMessage {
    direction: rtl;
    text-align: right;
    unicode-bidi: embed;
}

/* Better Arabic rendering */
p, li, div {
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25   # lowered from 0.35 → Arabic embeddings score lower
                               # 0.35 was filtering out too many valid Arabic results


# ========================= API KEY =========================
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
            entries = pickle.load(f)
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
    - "query: " prefix is required by multilingual-e5 at retrieval time.
    - Threshold set to 0.25 (tuned for Arabic — scores lower than English).
    - Simple deduplication on first-30-words fingerprint.
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
                continue

            text = entries[idx]["text"]

            # Deduplication: fingerprint by first 30 words
            normalized = " ".join(text.split()[:30])
            if normalized in seen:
                continue
            seen.add(normalized)

            results.append(text)

        return results

    except Exception:
        return []


# ========================= LLM CALL =========================
def get_text_response(system_prompt: str, user_message: str) -> str:
    """
    Stateless RAG call — sends system prompt + single augmented message.

    FIX from v1: removed full conversation history injection.
    The old approach sent raw past user messages alongside the current
    augmented message (context + question), creating inconsistent message
    shapes that confused Gemini. For a RAG use-case, each query is
    self-contained: the retrieved context already carries all needed info.
    Keeping it stateless makes retrieval behavior predictable and clean.
    """
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash-lite",
            system_instruction=system_prompt
        )
        response = model.generate_content(user_message)
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
# FIX from v1: completely rewrote the system prompt.
#
# The old prompt said "Be clear, concise, and ACADEMIC in tone" — this
# directly told the model to use formal language and overrode any chance
# of Egyptian dialect output. The model was simply following instructions.
#
# New prompt:
# 1. Explicitly teaches the model the 3 language modes it must handle.
# 2. Instructs it to detect Egyptian Arabic (عامية) vs formal (فصحى) vs English.
# 3. Maps each mode to the correct response register.
# 4. Removes the word "academic" which was killing the colloquial tone.

SYSTEM_PROMPT = """
أنت GIGO، مساعد ذكي وودود لطلاب منصة GIG في جامعة النيل.

## قواعد الإجابة:

### 1. اكتشاف لغة السؤال والرد بنفس اللغة:

- **إذا السؤال بالعامية المصرية** (زي: "إيه ده؟" / "عايز أعرف" / "ازاي"):
  → رد بالعامية المصرية الطبيعية، كأنك صاحب الطالب. خليك بسيط وودود.
  مثال: "أيوه يسطا، الموضوع ده بيقول إن..."

- **إذا السؤال بالعربية الفصحى** (رسمية، أكاديمية):
  → رد بالعربية الفصحى السليمة بأسلوب أكاديمي واضح.

- **إذا السؤال بالإنجليزية**:
  → رد بالإنجليزية فقط بشكل واضح ومباشر.

### 2. استخدام السياق (CONTEXT):
- السياق المقدم هو مصدرك الأساسي للإجابة.
- فكّر جيداً في السياق قبل الإجابة.
- إذا كانت الإجابة موجودة بوضوح في السياق، أجب مباشرة.
- إذا لم تجد إجابة واضحة في السياق:
  - بالعامية: "والله يا صديقي مش لاقي إجابة واضحة في البيانات، بس ممكن أساعدك بشكل عام."
  - بالفصحى: "لم أجد إجابة واضحة في البيانات المتاحة، لكن يمكنني المساعدة بشكل عام."
  - بالإنجليزية: "I couldn't find a clear answer in the available data, but I can help generally."

### 3. ممنوع:
- لا تخترع معلومات غير موجودة في السياق.
- لا تخلط اللغات في نفس الرد.
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
    st.markdown("### أهلاً! أنا جيجو، قولي أقدر أساعدك بإيه؟")

    lottie = load_lottie("https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json")
    if lottie:
        st_lottie(lottie, height=220)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display all previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    user_input = st.chat_input("Ask your question here... / اكتب سؤالك هنا...")

    if user_input:
        # Show user message immediately
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant Arabic context chunks
        relevant_chunks = retrieve_chunks(user_input)

        if relevant_chunks:
            context = "\n\n---\n\n".join(relevant_chunks)
        else:
            context = "لا توجد محتويات ذات صلة في قاعدة المعرفة. / No relevant content found in the knowledge base."

        # Build the augmented prompt — context + question in one clean message
        augmented_message = f"""CONTEXT (من قاعدة البيانات / from knowledge base):
{context}

QUESTION:
{user_input}"""

        # Generate response — stateless, no history injection
        with st.chat_message("assistant"):
            with st.spinner("🤔 بفكر..."):
                reply = get_text_response(SYSTEM_PROMPT, augmented_message)
                st.markdown( f"""
                            <div dir="rtl" style="text-align: right; line-height:2;">
                            {reply}
                            </div>
                            """,
                            unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "text": reply})

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ========================= PAGE 2: IMAGE SOLVER =========================
elif selected == "Image Solver":

    st.title("🖼️ Image Problem Solver")
    st.markdown("ارفع صورة وهنشرحلك المسألة")

    lottie = load_lottie("https://lottie.host/0db5a9c4-6c54-4c65-bb3d-66d7dc5a4d0e/0Tzv7FpC18.json")
    if lottie:
        st_lottie(lottie, height=220)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    VISION_PROMPT = """
حلل الصورة دي بعناية وشرح المسألة اللي فيها.

المطلوب منك:
- رد بالعامية المصرية البسيطة زي ما بتكلم صاحبك
- اشرح المسألة بطريقة سهلة ومفهومة
- لو فيه حل، اديه خطوة خطوة
- المستوى مناسب لطالب جامعي
"""

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة اللي رفعتها", use_column_width=True)

        if st.button("🔍 اشرحلي المسألة"):
            with st.spinner("⏳ بحلل الصورة..."):
                result = get_vision_response(VISION_PROMPT, image)
            st.success(result)
