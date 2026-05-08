import streamlit as st
import fitz  # pymupdf — better Arabic PDF extraction than PyPDF2
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import re
import google.generativeai as genai

# ========================= CONFIG =========================
DATA_PATH = "data/"
CHUNK_SIZE = 700       # raised from 400 → Arabic is denser, needs bigger chunks
CHUNK_OVERLAP = 120    # raised proportionally

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# ========================= API KEY =========================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ API Key not found. Add GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()


# ========================= TEXT EXTRACTION =========================
def extract_text(pdf_path: str) -> str:
    """
    Uses pymupdf (fitz) instead of PyPDF2.
    pymupdf handles Arabic RTL text, complex encoding, and diacritics correctly.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text("text")   # "text" mode preserves reading order
        if text.strip():
            pages_text.append(text)
    doc.close()
    return "\n".join(pages_text)


# ========================= TEXT CLEANING =========================
def clean_text(text: str) -> str:
    """
    Cleans extracted Arabic text.
    - Collapses excess whitespace
    - Removes hyphenated line-breaks (common in PDF extraction)
    - Removes repeated dots (table-of-contents artifacts)
    - Preserves Arabic punctuation characters
    """
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'\.{3,}', '', text)         # remove dotted lines
    text = re.sub(r'-\s+', '', text)           # fix hyphenated line breaks
    text = text.strip()
    return text


# ========================= CHUNKING (ARABIC-AWARE) =========================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Arabic-aware sentence splitter.

    FIX from v1: original regex only split on English punctuation (.!?)
    Arabic uses ؟ (Arabic question mark) and ، (Arabic comma) as major
    sentence boundaries. Without including these, the entire Arabic document
    was returned as one giant chunk, destroying retrieval quality.

    Now splits on: . ! ? ؟ ، — covering both Arabic and English text.
    """
    sentences = re.split(r'(?<=[.!?؟،])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # carry overlap from end of previous chunk for continuity
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ========================= MAIN PIPELINE =========================
st.title("📄 RAG Pipeline Builder")
st.write("🔄 Processing Arabic PDFs...")

all_chunks = []
all_sources = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        path = os.path.join(DATA_PATH, file)

        raw_text = extract_text(path)        # pymupdf extraction
        clean = clean_text(raw_text)         # Arabic-safe cleaning
        chunks = chunk_text(clean)           # Arabic-aware chunking

        all_chunks.extend(chunks)
        all_sources.extend([file] * len(chunks))

        st.write(f"✅ {file}: {len(chunks)} chunks")

st.write(f"📚 Total Arabic chunks: {len(all_chunks)}")

# ---------------------------------------------------------------
# NOTE: Translation step REMOVED.
# Your source files are already in Arabic (فصحى).
# Translating Arabic → Arabic via Gemini was wasting API calls and
# introducing noise/errors into the index. We embed directly.
# ---------------------------------------------------------------

# Build entries list directly from Arabic source chunks
entries = [
    {"text": chunk, "lang": "ar", "source": all_sources[i]}
    for i, chunk in enumerate(all_chunks)
]

st.write(f"🔤 Total entries to embed: {len(entries)}")


# ========================= EMBEDDINGS =========================
st.write("🔢 Encoding embeddings with multilingual-e5-large...")

# e5 models require the "passage: " prefix at index time
# and "query: " prefix at query time — this is critical for quality
texts_to_embed = [f"passage: {entry['text']}" for entry in entries]

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

embeddings = embed_model.encode(
    texts_to_embed,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True   # required for cosine similarity via dot product
)


# ========================= FAISS INDEX =========================
dim = embeddings.shape[1]

# IndexFlatIP = inner product (= cosine similarity when vectors are normalized)
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "index.faiss")


# ========================= SAVE CHUNKS =========================
with open("chunks.pkl", "wb") as f:
    pickle.dump(entries, f)


st.success("✅ RAG Pipeline built successfully!")
st.write(f"📦 Vectors: {index.ntotal} | Dimensions: {dim}")
