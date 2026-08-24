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

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="GIG Assistant",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>

html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}

.stChatMessage {
    direction: rtl;
    text-align: right;
    unicode-bidi: embed;
}

p, li, div {
    line-height: 2;
}

/* Source badges */
.source-badge {
    display: inline-block;
    font-size: 0.72em;
    padding: 3px 10px;
    border-radius: 12px;
    margin: 2px 4px;
    font-family: monospace;
    font-weight: 600;
}

.source-pdf {
    background: #dbeafe;
    color: #1e40af;
}

.source-web {
    background: #dcfce7;
    color: #166534;
}

/* Debug box */
.debug-box {
    background: #111827;
    color: #f9fafb;
    padding: 12px;
    border-radius: 10px;
    font-size: 0.8em;
    margin-top: 10px;
    overflow-x: auto;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# CONSTANTS
# ==========================================================
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

TOP_K_PDF = 3
TOP_K_WEB = 3

SIMILARITY_THRESHOLD = 0.30

# ==========================================================
# API KEY
# ==========================================================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

except Exception:
    st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()

# ==========================================================
# LOAD RAG
# ==========================================================
@st.cache_resource
def load_rag():

    try:
        # ==============================
        # LOAD INDICES
        # ==============================
        pdf_index = faiss.read_index("pdf_index.faiss")

        web_index = faiss.read_index("web_index.faiss")

        # ==============================
        # LOAD CHUNKS
        # ==============================
        with open("pdf_chunks.pkl", "rb") as f:
            pdf_entries = pickle.load(f)

        with open("web_chunks.pkl", "rb") as f:
            web_entries = pickle.load(f)

        # ==============================
        # EMBEDDING MODEL
        # ==============================
        embed_model = SentenceTransformer(
            EMBED_MODEL_NAME
        )

        return (
            pdf_index,
            web_index,
            pdf_entries,
            web_entries,
            embed_model
        )

    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        st.stop()

(
    pdf_index,
    web_index,
    pdf_entries,
    web_entries,
    embed_model
) = load_rag()

# ==========================================================
# SOURCE LABEL
# ==========================================================
def format_source_label(source: str):

    if source.startswith("http"):

        domain = source.split("/")[2]

        return f"🌐 {domain}"

    return f"📄 {source}"

# ==========================================================
# RETRIEVAL
# ==========================================================
def retrieve_chunks(query: str):

    try:

        # ==========================================
        # QUERY EMBEDDING
        # ==========================================
        query_embedding = embed_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True
        ).astype("float32")

        # ==========================================
        # SEARCH PDF INDEX
        # ==========================================
        pdf_scores, pdf_indices = pdf_index.search(
            query_embedding,
            TOP_K_PDF
        )

        # ==========================================
        # SEARCH WEB INDEX
        # ==========================================
        web_scores, web_indices = web_index.search(
            query_embedding,
            TOP_K_WEB
        )

        results = []

        seen = set()

        # ==========================================
        # PDF RESULTS
        # ==========================================
        for score, idx in zip(
            pdf_scores[0],
            pdf_indices[0]
        ):

            if idx == -1:
                continue

            if score < SIMILARITY_THRESHOLD:
                continue

            entry = pdf_entries[idx]

            text = entry["text"]

            # Deduplication
            fingerprint = " ".join(
                text.split()[:30]
            )

            if fingerprint in seen:
                continue

            seen.add(fingerprint)

            results.append({
                "score": float(score),
                "text": text,
                "source": entry["source"],
                "type": "pdf"
            })

        # ==========================================
        # WEBSITE RESULTS
        # ==========================================
        for score, idx in zip(
            web_scores[0],
            web_indices[0]
        ):

            if idx == -1:
                continue

            if score < SIMILARITY_THRESHOLD:
                continue

            entry = web_entries[idx]

            text = entry["text"]

            fingerprint = " ".join(
                text.split()[:30]
            )

            if fingerprint in seen:
                continue

            seen.add(fingerprint)

            # WEBSITE BOOST
            adjusted_score = (
                float(score)
                + entry.get("boost", 0.03)
            )

            results.append({
                "score": adjusted_score,
                "text": text,
                "source": entry["source"],
                "type": "website"
            })

        # ==========================================
        # GLOBAL RERANK
        # ==========================================
        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        return results

    except Exception as e:

        st.error(f"❌ Retrieval Error: {e}")

        return []

# ==========================================================
# TEXT RESPONSE
# ==========================================================
def get_text_response(
    system_prompt: str,
    user_message: str
):

    try:

        model = genai.GenerativeModel(
            "gemini-2.5-flash-lite",
            system_instruction=system_prompt
        )

        response = model.generate_content(
            user_message
        )

        return response.text.strip()

    except Exception as e:

        return f"❌ Error generating response: {e}"

# ==========================================================
# VISION RESPONSE
# ==========================================================
def get_vision_response(prompt: str, image):

    try:

        model = genai.GenerativeModel(
            "gemini-2.5-flash-lite"
        )

        response = model.generate_content(
            [prompt, image]
        )

        return response.text.strip()

    except Exception as e:

        return f"❌ Error analyzing image: {e}"

# ==========================================================
# LOTTIE
# ==========================================================
def load_lottie(url: str):

    try:
        return requests.get(url).json()

    except Exception:
        return None

# ==========================================================
# SYSTEM PROMPT
# ==========================================================
SYSTEM_PROMPT = """
أنت GIGO، مساعد ذكي وودود لطلاب منصة GIG في جامعة النيل.

==========================
قواعد الرد
==========================

1) اكتشاف اللغة تلقائياً:

- لو المستخدم بيتكلم بالعامية المصرية:
  رد بالعامية المصرية الطبيعية والبسيطة.

- لو المستخدم بيتكلم بالعربية الفصحى:
  رد بالعربية الفصحى.

- لو المستخدم بيتكلم بالإنجليزية:
  رد بالإنجليزية فقط.

==========================
استخدام السياق
==========================

السياق يأتي من:
- ملفات PDF الرسمية
- الموقع الإلكتروني الرسمي

كلاهما مصادر موثوقة.

- استخدم فقط المعلومات الموجودة في السياق.
- لا تخترع معلومات غير موجودة.
- إذا لم تجد إجابة واضحة:

عامية:
"والله مش لاقي إجابة واضحة في البيانات المتاحة، بس ممكن أساعدك بشكل عام."

فصحى:
"لم أجد إجابة واضحة في البيانات المتاحة، لكن يمكنني المساعدة بشكل عام."

English:
"I couldn't find a clear answer in the available data, but I can help generally."

==========================
أسلوب الرد
==========================

- كن واضحاً ومباشراً.
- لا تخلط اللغات.
- لا تكرر نفس الجملة كثيراً.
- استخدم تنسيقاً جيداً.
"""

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    selected = option_menu(
        "GIG Assistant",
        ["Chat Assistant", "Image Solver"],
        menu_icon="cpu",
        icons=["chat-dots-fill", "image-fill"],
        default_index=0
    )

# ==========================================================
# CHAT PAGE
# ==========================================================
if selected == "Chat Assistant":

    st.title("🤖 GIGO")

    st.markdown(
        "### Hi I'm GIGO, how can I help you"
    )

    # ======================================================
    # LOTTIE
    # ======================================================
    lottie = load_lottie(
        "https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json"
    )

    if lottie:
        st_lottie(lottie, height=220)

    # ======================================================
    # SESSION STATE
    # ======================================================
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ======================================================
    # DISPLAY HISTORY
    # ======================================================
    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):

            st.markdown(
                f"""
                <div dir="rtl"
                style="text-align:right;line-height:2;">
                {msg["text"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==============================================
            # SOURCES
            # ==============================================
            if msg.get("sources"):

                badges = ""

                for src in msg["sources"]:

                    css = (
                        "source-web"
                        if src.startswith("http")
                        else "source-pdf"
                    )

                    label = format_source_label(src)

                    badges += (
                        f'<span class="source-badge {css}">'
                        f'{label}'
                        f'</span>'
                    )

                st.markdown(
                    f"""
                    <div style="margin-top:8px;">
                    {badges}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ======================================================
    # USER INPUT
    # ======================================================
    user_input = st.chat_input(
        "حابب تسأل عن ايه؟"
    )

    # ======================================================
    # HANDLE INPUT
    # ======================================================
    if user_input:

        # ==============================================
        # USER MESSAGE
        # ==============================================
        st.session_state.chat_history.append({
            "role": "user",
            "text": user_input
        })

        with st.chat_message("user"):

            st.markdown(
                f"""
                <div dir="rtl"
                style="text-align:right;line-height:2;">
                {user_input}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ==============================================
        # RETRIEVE
        # ==============================================
        retrieved = retrieve_chunks(
            user_input
        )

        # ==============================================
        # CONTEXT
        # ==============================================
        if retrieved:

            context_parts = []

            unique_sources = []

            debug_text = ""

            for r in retrieved:

                source_label = format_source_label(
                    r["source"]
                )

                context_parts.append(
                    f"[المصدر: {source_label}]\n{r['text']}"
                )

                if r["source"] not in unique_sources:
                    unique_sources.append(r["source"])

                debug_text += (
                    f"\nTYPE: {r['type']}"
                    f"\nSCORE: {r['score']:.4f}"
                    f"\nSOURCE: {r['source']}"
                    f"\n{'-'*40}"
                )

            context = "\n\n---\n\n".join(
                context_parts
            )

        else:

            context = (
                "لا توجد معلومات ذات صلة."
            )

            unique_sources = []

            debug_text = "No chunks retrieved."

        # ==============================================
        # AUGMENTED PROMPT
        # ==============================================
        augmented_message = f"""
CONTEXT:
{context}

QUESTION:
{user_input}
"""

        # ==============================================
        # ASSISTANT RESPONSE
        # ==============================================
        with st.chat_message("assistant"):

            with st.spinner("بفكر..."):

                reply = get_text_response(
                    SYSTEM_PROMPT,
                    augmented_message
                )

            st.markdown(
                f"""
                <div dir="rtl"
                style="text-align:right;line-height:2;">
                {reply}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==========================================
            # SOURCE BADGES
            # ==========================================
            if unique_sources:

                badges = ""

                for src in unique_sources:

                    css = (
                        "source-web"
                        if src.startswith("http")
                        else "source-pdf"
                    )

                    label = format_source_label(src)

                    badges += (
                        f'<span class="source-badge {css}">'
                        f'{label}'
                        f'</span>'
                    )

                st.markdown(
                    f"""
                    <div style="margin-top:8px;">
                    <small>المصادر:</small><br>
                    {badges}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ==========================================
            # DEBUG SECTION
            # ==========================================
            with st.expander(
                "🔍 Retrieval Debug"
            ):

                st.markdown(
                    f"""
                    <div class="debug-box">
                    {debug_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ==============================================
        # SAVE HISTORY
        # ==============================================
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": reply,
            "sources": unique_sources
        })

    # ======================================================
    # CLEAR CHAT
    # ======================================================
    if st.session_state.chat_history:

        if st.button("🗑️ Clear Chat"):

            st.session_state.chat_history = []

            st.rerun()

# ==========================================================
# IMAGE SOLVER
# ==========================================================
elif selected == "Image Solver":

    st.title("🖼️ Image Problem Solver")

    st.markdown(
        "ارفع صورة وهنشرحلك المسألة"
    )

    lottie = load_lottie(
        "https://lottie.host/0db5a9c4-6c54-4c65-bb3d-66d7dc5a4d0e/0Tzv7FpC18.json"
    )

    if lottie:
        st_lottie(lottie, height=220)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    VISION_PROMPT = """
حلل الصورة بعناية.

- اشرح بطريقة بسيطة
- استخدم العامية المصرية
- لو فيه حل اشرح خطوة خطوة
"""

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="الصورة المرفوعة",
            use_column_width=True
        )

        if st.button("🔍 اشرحلي المسألة"):

            with st.spinner("⏳ بحلل الصورة..."):

                result = get_vision_response(
                    VISION_PROMPT,
                    image
                )

            st.success(result)
