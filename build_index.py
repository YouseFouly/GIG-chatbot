import streamlit as st
import fitz
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import re
import hashlib
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# ==========================================================
# CONFIG
# ==========================================================
DATA_PATH = "data/"

PDF_CHUNK_SIZE = 850
WEB_CHUNK_SIZE = 450

CHUNK_OVERLAP = 120

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

WEBSITE_BOOST = 0.03

WEBSITES_TO_SCRAPE = [
    "https://iecc.nu.edu.eg/gig-skill-boost-program/#course-frontend",
    "https://iecc.nu.edu.eg/gig-skill-boost-program/#courses",
    "https://iecc.nu.edu.eg/gig-skill-boost-program/#course-uiux",
    "https://iecc.nu.edu.eg/gig-skill-boost-program/#course-mobile",
    "https://iecc.nu.edu.eg/gig-skill-boost-program/#course-cs",
    "https://iecc.nu.edu.eg/gig-skill-boost-program/#course-marketing",
    "https://iecc.nu.edu.eg/gig-skill-boost-program/#course-sales",
]

NOISE_TAGS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "noscript",
    "iframe",
    "svg",
    "button",
    "meta",
    "link",
    "form",
]

# ==========================================================
# API KEY
# ==========================================================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ GOOGLE_API_KEY missing from Streamlit secrets.")
    st.stop()

# ==========================================================
# PDF EXTRACTION
# ==========================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        text = page.get_text("text")

        if text.strip():
            pages.append(text)

    doc.close()

    return "\n".join(pages)

# ==========================================================
# WEBSITE EXTRACTION
# ==========================================================
def extract_text_from_url(url: str) -> str:

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ar,en;q=0.9",
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove useless tags
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # ==================================================
        # PRIORITY CONTENT EXTRACTION
        # ==================================================
        # Try extracting only meaningful containers
        # instead of entire DOM text
        # ==================================================

        content = None

        selectors = [
            "main",
            "article",
            ".elementor-widget-container",
            ".elementor-section",
            ".content",
            ".container",
            "#content",
        ]

        for selector in selectors:

            found = soup.select(selector)

            if found:
                content = "\n".join(
                    [x.get_text(separator="\n", strip=True)
                     for x in found]
                )

                if len(content) > 500:
                    break

        # fallback
        if not content:
            content = soup.get_text(separator="\n", strip=True)

        return content

    except Exception as e:
        st.warning(f"⚠️ Failed scraping {url}: {e}")
        return ""

# ==========================================================
# CLEANING
# ==========================================================
def clean_text(text: str) -> str:

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\.{3,}', '', text)

    text = re.sub(r'-\s+', '', text)

    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    return text.strip()

# ==========================================================
# FINGERPRINTING
# ==========================================================
def compute_fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_fingerprints(path="fingerprints.pkl"):

    if os.path.exists(path):

        with open(path, "rb") as f:
            return pickle.load(f)

    return {}

def save_fingerprints(fingerprints, path="fingerprints.pkl"):

    with open(path, "wb") as f:
        pickle.dump(fingerprints, f)

# ==========================================================
# PDF CHUNKING
# ==========================================================
def chunk_pdf_text(
    text: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):

    sentences = re.split(r'(?<=[.!?؟])\s+', text)

    chunks = []

    current = ""

    for sentence in sentences:

        if len(current) + len(sentence) <= chunk_size:
            current += " " + sentence

        else:

            if current.strip():
                chunks.append(current.strip())

            current = current[-overlap:] + " " + sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks

# ==========================================================
# WEBSITE CHUNKING
# ==========================================================
def chunk_web_text(
    text: str,
    chunk_size: int = WEB_CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):

    # Websites work better with paragraph chunking
    paragraphs = re.split(r'\n+', text)

    paragraphs = [
        p.strip()
        for p in paragraphs
        if len(p.strip()) > 40
    ]

    chunks = []

    current = ""

    for para in paragraphs:

        if len(current) + len(para) <= chunk_size:
            current += "\n" + para

        else:

            if current.strip():
                chunks.append(current.strip())

            current = current[-overlap:] + "\n" + para

    if current.strip():
        chunks.append(current.strip())

    return chunks

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("📄 Advanced Hybrid Arabic RAG Builder")

st.info(
    "Indexes PDFs + live websites with "
    "Arabic-aware chunking and optimized retrieval."
)

# ==========================================================
# STORAGE
# ==========================================================
all_entries = []

fingerprints = load_fingerprints()

fingerprints_updated = False

# ==========================================================
# STEP 1 — PDFs
# ==========================================================
st.subheader("📁 Step 1 — Processing PDFs")

for file in os.listdir(DATA_PATH):

    if not file.endswith(".pdf"):
        continue

    path = os.path.join(DATA_PATH, file)

    raw_text = extract_text_from_pdf(path)

    if not raw_text.strip():
        continue

    fingerprint = compute_fingerprint(raw_text)

    if fingerprints.get(file) == fingerprint:
        st.write(f"⏭️ {file} unchanged")

    else:
        st.write(f"🆕 {file} updated/new")

        fingerprints[file] = fingerprint

        fingerprints_updated = True

    clean = clean_text(raw_text)

    chunks = chunk_pdf_text(clean)

    for chunk in chunks:

        all_entries.append({
            "text": chunk,
            "source": file,
            "type": "pdf",
            "boost": 0.0,
        })

    st.write(f"✅ {file}: {len(chunks)} chunks")

# ==========================================================
# STEP 2 — Websites
# ==========================================================
st.subheader("🌐 Step 2 — Scraping Websites")

for url in WEBSITES_TO_SCRAPE:

    st.write(f"🔍 Scraping: {url}")

    raw_text = extract_text_from_url(url)

    if not raw_text.strip():

        st.warning(f"⚠️ No content extracted from {url}")

        continue

    fingerprint = compute_fingerprint(raw_text)

    if fingerprints.get(url) == fingerprint:

        st.write(f"⏭️ Website unchanged")

    else:

        if fingerprints.get(url):
            st.success("🔄 Website content changed")

        else:
            st.write("🆕 First indexing")

        fingerprints[url] = fingerprint

        fingerprints_updated = True

    clean = clean_text(raw_text)

    chunks = chunk_web_text(clean)

    for chunk in chunks:

        all_entries.append({
            "text": chunk,
            "source": url,
            "type": "website",
            "boost": WEBSITE_BOOST,
        })

    st.write(f"✅ {len(chunks)} chunks extracted")

# ==========================================================
# STEP 3 — EMBEDDINGS
# ==========================================================
st.subheader("🔢 Step 3 — Embedding")

texts_to_embed = [
    f"passage: {entry['text']}"
    for entry in all_entries
]

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

embeddings = embed_model.encode(
    texts_to_embed,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# ==========================================================
# STEP 4 — SEPARATE INDICES
# ==========================================================
st.subheader("🗂️ Step 4 — Building FAISS Indices")

pdf_embeddings = []
web_embeddings = []

pdf_entries = []
web_entries = []

for i, entry in enumerate(all_entries):

    if entry["type"] == "pdf":
        pdf_embeddings.append(embeddings[i])
        pdf_entries.append(entry)

    else:
        web_embeddings.append(embeddings[i])
        web_entries.append(entry)

pdf_embeddings = np.array(pdf_embeddings).astype("float32")
web_embeddings = np.array(web_embeddings).astype("float32")

dim = embeddings.shape[1]

# ==========================================================
# PDF INDEX
# ==========================================================
pdf_index = faiss.IndexFlatIP(dim)

if len(pdf_embeddings) > 0:
    pdf_index.add(pdf_embeddings)

# ==========================================================
# WEBSITE INDEX
# ==========================================================
web_index = faiss.IndexFlatIP(dim)

if len(web_embeddings) > 0:
    web_index.add(web_embeddings)

# ==========================================================
# SAVE EVERYTHING
# ==========================================================
faiss.write_index(pdf_index, "pdf_index.faiss")

faiss.write_index(web_index, "web_index.faiss")

with open("pdf_chunks.pkl", "wb") as f:
    pickle.dump(pdf_entries, f)

with open("web_chunks.pkl", "wb") as f:
    pickle.dump(web_entries, f)

if fingerprints_updated:
    save_fingerprints(fingerprints)

# ==========================================================
# DEBUGGING STATISTICS
# ==========================================================
pdf_count = len(pdf_entries)
web_count = len(web_entries)

st.subheader("📊 Statistics")

st.write(f"📄 PDF chunks: {pdf_count}")

st.write(f"🌐 Website chunks: {web_count}")

st.write(f"🔤 Total chunks: {len(all_entries)}")

# ==========================================================
# DONE
# ==========================================================
st.success("✅ Advanced RAG pipeline built successfully!")

st.info(
    "Your system now includes:\n"
    "- Separate PDF & Website FAISS indices\n"
    "- Website retrieval boosting\n"
    "- Arabic-aware chunking\n"
    "- Better website extraction\n"
    "- Smart website paragraph chunking\n"
    "- Change detection\n"
    "- Hybrid indexing architecture"
)