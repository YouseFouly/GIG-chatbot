import PyPDF2
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

DATA_PATH = "data/"

def extract_text(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

model = SentenceTransformer("all-MiniLM-L6-v2")

all_chunks = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        text = extract_text(os.path.join(DATA_PATH, file))
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

embeddings = model.encode(all_chunks)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

faiss.write_index(index, "index.faiss")

with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ DONE")