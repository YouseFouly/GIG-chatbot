import streamlit as st
import PyPDF2
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import re
import google.generativeai as genai

# ========================= CONFIG =========================
DATA_PATH = "data/"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# ========================= API KEY (STREAMLIT SAFE) =========================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("❌ API Key not found. Add GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()


# ========================= TEXT EXTRACTION =========================
def extract_text(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


# ========================= TEXT CLEANING =========================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '', text)
    text = re.sub(r'-\s+', '', text)
    return text.strip()


# ========================= CHUNKING =========================
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ========================= TRANSLATION =========================
def translate_chunks_to_arabic(chunks):
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    translated = []

    st.write(f"🌐 Translating {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        try:
            prompt = f"Translate to Arabic only:\n\n{chunk}"
            response = model.generate_content(prompt)
            translated.append(response.text.strip())

        except Exception as e:
            st.warning(f"⚠️ Failed chunk {i}: {e}")
            translated.append("")

    return translated


# ========================= MAIN PIPELINE =========================
st.title("📄 RAG Pipeline Builder")

st.write("🔄 Processing PDFs...")

all_chunks = []
all_sources = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        path = os.path.join(DATA_PATH, file)

        raw_text = extract_text(path)
        clean = clean_text(raw_text)
        chunks = chunk_text(clean)

        all_chunks.extend(chunks)
        all_sources.extend([file] * len(chunks))

        st.write(f"✅ {file}: {len(chunks)} chunks")

st.write(f"📚 Total English chunks: {len(all_chunks)}")


# ========================= TRANSLATION =========================
arabic_chunks = translate_chunks_to_arabic(all_chunks)


# ========================= BILINGUAL DATA =========================
bilingual_entries = []

for i, (en_chunk, ar_chunk) in enumerate(zip(all_chunks, arabic_chunks)):
    bilingual_entries.append({
        "text": en_chunk,
        "lang": "en",
        "source": all_sources[i]
    })

    if ar_chunk:
        bilingual_entries.append({
            "text": ar_chunk,
            "lang": "ar",
            "source": all_sources[i]
        })


st.write(f"🔤 Total bilingual entries: {len(bilingual_entries)}")


# ========================= EMBEDDINGS =========================
st.write("🔢 Encoding embeddings...")

texts_to_embed = [
    f"passage: {entry['text']}" for entry in bilingual_entries
]

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

embeddings = embed_model.encode(
    texts_to_embed,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ========================= FAISS INDEX =========================
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "index.faiss")


# ========================= SAVE DATA =========================
with open("chunks.pkl", "wb") as f:
    pickle.dump(bilingual_entries, f)


st.success("✅ RAG Pipeline built successfully!")
st.write(f"📦 Vectors: {index.ntotal} | Dim: {dim}")
