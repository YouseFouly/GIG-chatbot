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
.stChatMessage {
    direction: rtl;
    text-align: right;
    unicode-bidi: embed;
}
p, li, div {
    line-height: 2;
}
/* Source badge styling */
.source-badge {
    display: inline-block;
    font-size: 0.72em;
    padding: 2px 8px;
    border-radius: 12px;
    margin: 2px 3px;
    font-family: monospace;
}
.source-pdf  { background: #dbeafe; color: #1e40af; }
.source-web  { background: #dcfce7; color: #166534; }
</style>
""", unsafe_allow_html=True)

EMBED_MODEL_NAME    = "intfloat/multilingual-e5-large"
TOP_K               = 5
SIMILARITY_THRESHOLD = 0.25

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
def retrieve_chunks(query: str, k: int = TOP_K) -> list[dict]:
    """
    Returns list of dicts: {"text": ..., "source": ...}
    instead of plain strings.

    WHY return source alongside text?
    Now that we have both PDFs and websites in the index, we want to:
    1. Tell the LLM WHERE each piece of context came from (PDF vs website).
    2. Show the user a source badge in the UI so they know if the answer
       came from the official document or the live website.

    This is a small change to the return type but enables full source tracing.
    """
    try:
        query_embedding = embed_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(query_embedding, k)

        results = []
        seen    = set()

        for score, idx in zip(scores[0], indices[0]):
            if score < SIMILARITY_THRESHOLD:
                continue

            entry = entries[idx]
            text  = entry["text"]

            # Deduplication fingerprint
            normalized = " ".join(text.split()[:30])
            if normalized in seen:
                continue
            seen.add(normalized)

            results.append({
                "text":   text,
                "source": entry.get("source", "unknown"),
            })

        return results

    except Exception:
        return []


# ===========================================================
# SOURCE LABEL HELPER                                 [NEW]
# ===========================================================
def format_source_label(source: str) -> str:
    """
    Returns a human-readable label for the source.
    - PDF filenames   → "📄 وثيقة: filename.pdf"
    - URLs            → "🌐 موقع: domain.com"
    """
    if source.startswith("http"):
        domain = source.split("/")[2]
        return f"🌐 {domain}"
    return f"📄 {source}"


# ========================= LLM CALL =========================
def get_text_response(system_prompt: str, user_message: str) -> str:
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
# Updated to mention that context may come from BOTH the official
# PDF documents AND the live website — so the model knows to treat
# both equally as authoritative sources.

SYSTEM_PROMPT = """
أنت GIGO، مساعد ذكي وودود لطلاب منصة GIG في جامعة النيل.

## قواعد الإجابة:

### 1. اكتشاف لغة السؤال والرد بنفس اللغة:

- **إذا السؤال بالعامية المصرية** (زي: "إيه ده؟" / "عايز أعرف" / "ازاي"):
  → رد بالعامية المصرية الطبيعية، كأنك صاحب الطالب. خليك بسيط وودود.
  مثال: "أيوه يسطا، الموضوع ده بيقول إن..."

- **إذا السؤال بالعربية الفصحى** (رسمية، أكاديمية):
  → رد بالعربية الفصحى السليمة بأسلوب واضح.

- **إذا السؤال بالإنجليزية**:
  → رد بالإنجليزية فقط بشكل واضح ومباشر.

### 2. استخدام السياق (CONTEXT):
- السياق المقدم مصدره إما وثائق PDF الرسمية أو الموقع الإلكتروني للبرنامج — كلاهما مصادر موثوقة.
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
    st.markdown("### Hi I'm GIGO, how can I help you")

    lottie = load_lottie("https://lottie.host/2fd251ba-a67b-4ea8-b9ea-ae3b1b5425f5/7AUv2Ddn0H.json")
    if lottie:
        st_lottie(lottie, height=220)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display all previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(
                f'<div dir="rtl" style="text-align:right;line-height:2;">{msg["text"]}</div>',
                unsafe_allow_html=True
            )
            # NEW: show source badges if available
            if msg.get("sources"):
                badge_html = ""
                for src in msg["sources"]:
                    css_class = "source-web" if src.startswith("http") else "source-pdf"
                    label     = format_source_label(src)
                    badge_html += f'<span class="source-badge {css_class}">{label}</span>'
                st.markdown(
                    f'<div style="margin-top:6px;direction:rtl;">{badge_html}</div>',
                    unsafe_allow_html=True
                )

    user_input = st.chat_input("حابب تسأل عن ايه..")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(
                f'<div dir="rtl" style="text-align:right;line-height:2;">{user_input}</div>',
                unsafe_allow_html=True
            )

        # Retrieve chunks — now returns dicts with text + source
        retrieved = retrieve_chunks(user_input)

        if retrieved:
            # Build context string, telling the LLM the source of each chunk
            context_parts = []
            for r in retrieved:
                source_label = format_source_label(r["source"])
                context_parts.append(f"[المصدر: {source_label}]\n{r['text']}")
            context = "\n\n---\n\n".join(context_parts)

            # Collect unique sources for the badge display
            unique_sources = list(dict.fromkeys(r["source"] for r in retrieved))
        else:
            context        = "لا توجد محتويات ذات صلة في قاعدة المعرفة."
            unique_sources = []

        augmented_message = f"""CONTEXT (من قاعدة البيانات / from knowledge base):
{context}

QUESTION:
{user_input}"""

        with st.chat_message("assistant"):
            with st.spinner("بفكر..."):
                reply = get_text_response(SYSTEM_PROMPT, augmented_message)

            st.markdown(
                f'<div dir="rtl" style="text-align:right;line-height:2;">{reply}</div>',
                unsafe_allow_html=True
            )

            # NEW: show source badges under the answer
            if unique_sources:
                badge_html = ""
                for src in unique_sources:
                    css_class = "source-web" if src.startswith("http") else "source-pdf"
                    label     = format_source_label(src)
                    badge_html += f'<span class="source-badge {css_class}">{label}</span>'
                st.markdown(
                    f'<div style="margin-top:6px;direction:rtl;"><small>المصادر: {badge_html}</small></div>',
                    unsafe_allow_html=True
                )

        st.session_state.chat_history.append({
            "role":    "assistant",
            "text":    reply,
            "sources": unique_sources,   # NEW: persist sources in history
        })

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
