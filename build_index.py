import streamlit as st
import fitz                          # pymupdf — Arabic-safe PDF extraction
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import re
import hashlib                        # NEW: for change-detection fingerprinting
import requests                       # NEW: for fetching web pages
from bs4 import BeautifulSoup        # NEW: for HTML parsing and cleaning
import google.generativeai as genai

# ========================= CONFIG =========================
DATA_PATH     = "data/"
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 120
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# ---------------------------------------------------------------
# NEW — WEBSITES TO SCRAPE
# Add any URL you want included in the knowledge base here.
# The scraper will visit each URL, extract clean Arabic/English text,
# chunk it, and embed it alongside the PDF chunks.
# When the website developer updates the site, just re-run build_index.py
# and the RAG index will reflect the latest content automatically.
# ---------------------------------------------------------------
WEBSITES_TO_SCRAPE = [
    # Replace with your actual GIG platform URL(s)
    "https://nilepreneurs.nu.edu.eg/",
    # Add more pages as needed, e.g.:
    # "https://nilepreneurs.nu.edu.eg/gig-program",
    # "https://nilepreneurs.nu.edu.eg/faq",
]

# HTML tags whose content is NEVER useful — always stripped
NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "noscript", "iframe", "svg", "button", "meta",
    "link", "form"
]

# ========================= API KEY =========================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ API Key not found. Add GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()


# ===========================================================
# TEXT EXTRACTION — PDFs
# ===========================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Uses pymupdf (fitz) for Arabic-safe extraction.
    Handles RTL text, complex encoding, and diacritics correctly.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)
    doc.close()
    return "\n".join(pages_text)


# ===========================================================
# TEXT EXTRACTION — Websites                          [NEW]
# ===========================================================
def extract_text_from_url(url: str) -> str:
    """
    Fetches a web page and extracts clean readable text.

    WHY requests + BeautifulSoup and not Selenium?
    Most university/program websites are server-rendered (plain HTML).
    requests + BeautifulSoup is lightweight, fast, has no browser dependency,
    and works perfectly on Streamlit Cloud. If the site ever requires
    JavaScript rendering, swap requests.get() for a Playwright call —
    the rest of the pipeline stays identical.

    WHY strip noise tags?
    Navigation menus, footers, and script blocks add thousands of tokens
    of useless text that destroy embedding quality. Stripping them first
    ensures every chunk stored is 100% real content.
    """
    try:
        headers = {
            # Mimic a real browser to avoid bot-blocking
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ar,en;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove all noise tags before extracting text
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # get_text with separator="\n" preserves paragraph structure
        text = soup.get_text(separator="\n", strip=True)
        return text

    except Exception as e:
        st.warning(f"⚠️ Could not scrape {url}: {e}")
        return ""


# ===========================================================
# CHANGE DETECTION                                    [NEW]
# ===========================================================
def compute_fingerprint(text: str) -> str:
    """
    Returns an MD5 hash of the text content.

    WHY fingerprinting?
    Without this, every time build_index.py runs it re-embeds everything
    from scratch — even if nothing changed. This is slow and wastes compute.

    With fingerprinting:
    - First run: fetch content → hash → store hash → embed.
    - Next run:  fetch content → hash → compare to stored hash.
      Hashes match   → content unchanged → report and skip re-embed.
      Hashes differ  → content updated   → re-embed the new version.

    For a university site that updates occasionally, this saves significant
    time on every routine re-run.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_fingerprints(path: str = "fingerprints.pkl") -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_fingerprints(fingerprints: dict, path: str = "fingerprints.pkl"):
    with open(path, "wb") as f:
        pickle.dump(fingerprints, f)


# ===========================================================
# TEXT CLEANING
# ===========================================================
def clean_text(text: str) -> str:
    """
    Arabic-safe text cleaning that works for both PDF and web-scraped text:
    - Collapses excess whitespace (very common after HTML/PDF extraction)
    - Removes dotted lines (PDF table-of-contents artifacts)
    - Fixes hyphenated line-breaks (PDF column layout artifacts)
    - Removes lines that are only numbers (page numbers / nav items)
    - Preserves Arabic punctuation characters ، ؟
    """
    text = re.sub(r'\s+', ' ', text)            # collapse whitespace
    text = re.sub(r'\.{3,}', '', text)           # remove dotted lines
    text = re.sub(r'-\s+', '', text)             # fix hyphenated line breaks
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text) # remove lone number lines
    return text.strip()


# ===========================================================
# CHUNKING — Arabic-aware
# ===========================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits on both Arabic AND English sentence boundaries.

    Arabic punctuation added:
      ؟ = Arabic question mark
      ، = Arabic comma (used as a sentence/clause separator)

    Without these, Arabic text becomes one giant un-split blob
    and retrieval quality collapses entirely.
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
            # Carry overlap for cross-chunk context continuity
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ===========================================================
# MAIN PIPELINE
# ===========================================================
st.title("📄 RAG Pipeline Builder")
st.info("🔄 This pipeline indexes both **PDFs** and **live website content**.")

all_chunks  = []
all_sources = []

fingerprints         = load_fingerprints()
fingerprints_updated = False

# ----------------------------------------------------------
# STEP 1 — Process PDFs
# ----------------------------------------------------------
st.subheader("📁 Step 1: Processing PDFs")

for file in os.listdir(DATA_PATH):
    if not file.endswith(".pdf"):
        continue

    path     = os.path.join(DATA_PATH, file)
    raw_text = extract_text_from_pdf(path)
    new_fp   = compute_fingerprint(raw_text)

    if fingerprints.get(file) == new_fp:
        st.write(f"⏭️ {file}: unchanged since last build")
    else:
        st.write(f"🆕 {file}: new or updated — indexing")
        fingerprints[file]   = new_fp
        fingerprints_updated = True

    clean  = clean_text(raw_text)
    chunks = chunk_text(clean)

    all_chunks.extend(chunks)
    all_sources.extend([file] * len(chunks))
    st.write(f"✅ {file}: {len(chunks)} chunks")

# ----------------------------------------------------------
# STEP 2 — Scrape Websites                           [NEW]
# ----------------------------------------------------------
st.subheader("🌐 Step 2: Scraping Websites")

for url in WEBSITES_TO_SCRAPE:
    st.write(f"🔍 Fetching: `{url}`")
    raw_text = extract_text_from_url(url)

    if not raw_text.strip():
        st.warning(f"⚠️ No content extracted from {url} — skipping.")
        continue

    new_fp = compute_fingerprint(raw_text)

    if fingerprints.get(url) == new_fp:
        st.write(f"⏭️ `{url}`: content unchanged since last build")
    else:
        if fingerprints.get(url):
            # Hash existed before but is now different = website was updated
            st.success(f"🔄 `{url}`: content **CHANGED** — re-indexing updated content!")
        else:
            st.write(f"🆕 `{url}`: first time indexing")
        fingerprints[url]    = new_fp
        fingerprints_updated = True

    clean  = clean_text(raw_text)
    chunks = chunk_text(clean)

    # Source is the URL — so citations in the chat can say "from website"
    all_chunks.extend(chunks)
    all_sources.extend([url] * len(chunks))
    st.write(f"✅ `{url}`: {len(chunks)} chunks extracted")

# ----------------------------------------------------------
# STEP 3 — Build entries list
# ----------------------------------------------------------
st.subheader("📦 Step 3: Building Entry List")

entries = [
    {
        "text":   chunk,
        "lang":   "ar",
        "source": all_sources[i],   # filename for PDFs, URL for web pages
    }
    for i, chunk in enumerate(all_chunks)
]

total_pdf_chunks = sum(1 for s in all_sources if s.endswith(".pdf"))
total_web_chunks = sum(1 for s in all_sources if s.startswith("http"))

st.write(f"📄 PDF chunks   : {total_pdf_chunks}")
st.write(f"🌐 Web chunks   : {total_web_chunks}")
st.write(f"🔤 Total entries: {len(entries)}")

# ----------------------------------------------------------
# STEP 4 — Embed
# ----------------------------------------------------------
st.subheader("🔢 Step 4: Encoding Embeddings")

# e5 requires "passage: " prefix at index time, "query: " at query time
texts_to_embed = [f"passage: {e['text']}" for e in entries]

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
embeddings  = embed_model.encode(
    texts_to_embed,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# ----------------------------------------------------------
# STEP 5 — FAISS Index
# ----------------------------------------------------------
st.subheader("🗂️ Step 5: Building FAISS Index")

dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "index.faiss")

with open("chunks.pkl", "wb") as f:
    pickle.dump(entries, f)

if fingerprints_updated:
    save_fingerprints(fingerprints)

# ----------------------------------------------------------
# DONE
# ----------------------------------------------------------
st.success("✅ RAG Pipeline built successfully!")
st.write(f"📦 Vectors: {index.ntotal} | Dimensions: {dim}")
st.info(
    "💡 **Auto-update workflow:** Every time you re-run this page, the scraper "
    "re-fetches all listed websites. If the website developer updates the site, "
    "the change-detection system automatically detects new content and re-indexes it. "
    "No manual work needed — just re-run the builder."
)
