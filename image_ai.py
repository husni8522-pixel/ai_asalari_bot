import os
import pickle
import numpy as np
import faiss
import json
from openai import OpenAI

IMAGE_DIR = "images"
INDEX_FILE = "image_faiss.index"
META_FILE = "image_meta.pkl"

client = OpenAI()

# ===============================
# 🔹 Sinonim bazani yuklash
# ===============================
if os.path.exists("image_synonyms_big.json"):
    with open("image_synonyms_big.json", "r", encoding="utf-8") as f:
        IMAGE_KB = json.load(f)

elif os.path.exists("image_synonyms.json"):
    with open("image_synonyms.json", "r", encoding="utf-8") as f:
        IMAGE_KB = json.load(f)

else:
    IMAGE_KB = {}

# ===============================
# 🔹 Savolni sinonim bilan kengaytirish
# ===============================
def expand_question(question):

    q = question.lower()

    for key, data in IMAGE_KB.items():

        if not isinstance(data, dict):
            continue

        for lang_words in data.values():

            if not isinstance(lang_words, list):
                continue

            for word in lang_words:

                if word.lower() in q:

                    # barcha sinonimlarni qo‘shamiz
                    expanded = " ".join(lang_words)

                    return q + " " + expanded

    return q


# ===============================
# 🔹 Description builder
# ===============================
def build_description(base_name, folder):

    data = IMAGE_KB.get(base_name)

    words = []

    if data and isinstance(data, dict):

        for lang in data:
            if isinstance(data[lang], list):
                words.extend(data[lang])

    words.append(base_name)
    words.append(folder)

    return " ".join(words)


# ===============================
# 🔹 Build Image Index
# ===============================
def build_image_index():

    if not os.path.exists(IMAGE_DIR):
        print("❌ images papka topilmadi")
        return

    texts = []
    paths = []

    for root, dirs, files in os.walk(IMAGE_DIR):

        for file in files:

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, IMAGE_DIR)

            name = os.path.splitext(file)[0].lower()
            folder = os.path.basename(root).lower()

            description = build_description(name, folder)

            texts.append(description)
            paths.append(relative_path)

    if not texts:
        print("❌ Rasm topilmadi")
        return

    embeddings = []

    for text in texts:

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        embeddings.append(emb)

    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    pickle.dump(paths, open(META_FILE, "wb"))

    print("🖼 Image FAISS index created")


# ===============================
# 🔹 Image Search
# ===============================
def find_images_for_question(question, threshold=0.35):

    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        build_image_index()

    if not os.path.exists(INDEX_FILE):
        return []

    index = faiss.read_index(INDEX_FILE)
    meta = pickle.load(open(META_FILE, "rb"))

    # 🔥 Savolni sinonim bilan kengaytiramiz
    question = expand_question(question)

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    q_emb = np.array([q_emb]).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, 5)

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        if score >= threshold:
            results.append(os.path.join(IMAGE_DIR, meta[idx]))

    return results

