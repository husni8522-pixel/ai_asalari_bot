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
# 🔹 Sinonim bazasi
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
# 🔹 KEYWORD SEARCH
# ===============================

def keyword_search(question):

    q = question.lower()
    results = []

    for file in os.listdir(IMAGE_DIR):

        if not file.lower().endswith(("jpg","jpeg","png")):
            continue

        name = os.path.splitext(file)[0].lower()

        # 1️⃣ file nomi
        if name in q:
            results.append(os.path.join(IMAGE_DIR,file))
            continue

        # 2️⃣ sinonim
        data = IMAGE_KB.get(name)

        if data:
            for words in data.values():
                for w in words:
                    if w.lower() in q:
                        results.append(os.path.join(IMAGE_DIR,file))

    return list(set(results))


# ===============================
# 🔹 DESCRIPTION
# ===============================

def build_description(name):

    words=[name]

    data=IMAGE_KB.get(name)

    if data:
        for w in data.values():
            words.extend(w)

    return " ".join(words)


# ===============================
# 🔹 BUILD INDEX
# ===============================

def build_image_index():

    texts=[]
    paths=[]

    for file in os.listdir(IMAGE_DIR):

        if not file.lower().endswith(("jpg","jpeg","png")):
            continue

        name=os.path.splitext(file)[0].lower()

        desc=build_description(name)

        texts.append(desc)
        paths.append(os.path.join(IMAGE_DIR,file))

    embeddings=[]

    for t in texts:

        emb=client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        ).data[0].embedding

        embeddings.append(emb)

    vectors=np.array(embeddings).astype("float32")

    faiss.normalize_L2(vectors)

    index=faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)

    faiss.write_index(index,INDEX_FILE)

    pickle.dump(paths,open(META_FILE,"wb"))

    print("🖼 Image FAISS index created")


# ===============================
# 🔹 EMBEDDING SEARCH
# ===============================

def embedding_search(question):

    if not os.path.exists(INDEX_FILE):
        build_image_index()

    index=faiss.read_index(INDEX_FILE)

    meta=pickle.load(open(META_FILE,"rb"))

    emb=client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    vec=np.array([emb]).astype("float32")

    faiss.normalize_L2(vec)

    scores,ids=index.search(vec,5)

    results=[]

    for i in ids[0]:

        if i==-1:
            continue

        results.append(meta[i])

    return results


# ===============================
# 🔹 HYBRID SEARCH
# ===============================

def find_images_for_question(question):

    # 1️⃣ keyword search
    kw=keyword_search(question)

    if kw:
        return kw[:3]

    # 2️⃣ embedding search
    emb=embedding_search(question)

    return emb[:3]
