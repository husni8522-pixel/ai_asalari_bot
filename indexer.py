
# GLOBAL CACHE
INDEX = None
TEXTS = None
import os
import faiss
import pickle
import numpy as np
from config import *
from docx import Document
from pypdf import PdfReader


def read_file(p):
    try:
        if p.endswith(".docx"):
            return "\n".join(x.text for x in Document(p).paragraphs)
        if p.endswith(".pdf"):
            return "\n".join(pg.extract_text() or "" for pg in PdfReader(p).pages)
        if p.endswith(".txt"):
            with open(p, encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"❌ File read error: {p} -> {e}")
    return ""


def chunk_text(text, size=CHUNK_SIZE):
    if not text:
        return []
    return [text[i:i+size] for i in range(0, len(text), size)]


def build_index():
    print("♻️ INDEX YARATILYAPTI...")
    docs = []

    os.makedirs(DATA_DIR, exist_ok=True)

    for f in os.listdir(DATA_DIR):
        if f.endswith((".pdf", ".docx", ".txt")):
            text = read_file(os.path.join(DATA_DIR, f))

            for c in chunk_text(text):
                if len(c.strip()) > 80:   # 🔥 faqat minimal uzunlik
                    docs.append(c.strip())

    if not docs:
        print("❌ DATA papkada hujjat yo‘q")
        return

    vectors = []

    for i in range(0, len(docs), BATCH_SIZE):
        r = client.embeddings.create(
            model="text-embedding-3-small",
            input=docs[i:i+BATCH_SIZE]
        )
        vectors.extend([d.embedding for d in r.data])

    index = faiss.IndexFlatIP(len(vectors[0]))  # 🔥 IP (cosine) ishlatamiz
    vectors = np.array(vectors).astype("float32")

    # cosine similarity uchun normalize qilamiz
    faiss.normalize_L2(vectors)

    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    pickle.dump(docs, open(META_FILE, "wb"))

    print("✅ INDEX TAYYOR")


def index_invalid():
    return (
        not os.path.exists(INDEX_FILE)
        or not os.path.exists(META_FILE)
    )


def search_docs(q, threshold=0.65):

    try:

        if index_invalid():
            build_index()
            if index_invalid():
                return []

        index = faiss.read_index(INDEX_FILE)
        texts = pickle.load(open(META_FILE, "rb"))

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=[q]
        ).data[0].embedding

        emb = np.array([emb]).astype("float32")

        D, I = index.search(emb, TOP_K)

        results = []

        for dist, idx in zip(D[0], I[0]):

            if idx == -1:
                continue

            # L2 → similarity
            similarity = 1 / (1 + dist)

            if similarity >= threshold:
                results.append(texts[idx])

        return results

    except Exception as e:
        print("Search error:", e)
        return []

