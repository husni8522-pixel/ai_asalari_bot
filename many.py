import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from langdetect import detect
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters
)
from openai import OpenAI
from docx import Document
from pypdf import PdfReader

# ================== CONFIG ==================
DATA_DIR = "data"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"

CHUNK_SIZE = 1000
BATCH_SIZE = 32
TOP_K = 10
MAX_MEMORY = 5

# ================== LOAD ENV ==================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))

if not BOT_TOKEN or not OPENAI_KEY:
    raise RuntimeError("❌ .env faylda TOKEN yoki OPENAI KEY yo‘q")

client = OpenAI(api_key=OPENAI_KEY)

# ================== USER MEMORY ==================
user_memory = {}  # user_id -> [questions]

# ================== ASALARICHILIK MODULI ==================
ASALARI_TOPICS = {
    "kasallik": [
        "kasallik", "varroa", "nosema", "akarapidoz",
        "канa", "варроа", "болезнь",
        "disease", "mite"
    ],
    "parvarish": [
        "parvarish", "qishlatish", "uyacha", "ramka",
        "уход", "зимовка",
        "care", "wintering"
    ],
    "oziqlantirish": [
        "oziqlantirish", "shakar", "sirop", "kandi",
        "корм", "сироп",
        "feeding", "syrup"
    ],
    "ona_ari": [
        "ona ari", "qirolicha",
        "матка",
        "queen bee"
    ],
    "mahsulot": [
        "asal", "mum", "perga", "propolis",
        "мёд", "воск",
        "honey", "wax"
    ]
}

def is_asalari(text: str) -> bool:
    t = text.lower()
    for words in ASALARI_TOPICS.values():
        if any(w in t for w in words):
            return True
    return False

def detect_topic(text: str) -> str:
    t = text.lower()
    for topic, words in ASALARI_TOPICS.items():
        if any(w in t for w in words):
            return topic
    return "umumiy"

# ================== LANGUAGE ==================
def detect_lang(text):
    try:
        l = detect(text)
        if l.startswith("ru"):
            return "ru"
        if l.startswith("en"):
            return "en"
        return "uz"
    except:
        return "uz"

def translate(text, target):
    if detect_lang(text) == target:
        return text
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": f"Translate to {target}. Only translation."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return r.choices[0].message.content.strip()

# ================== FILE READ ==================
def read_file(path):
    if path.endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def chunk_text(text):
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

# ================== LOAD DOCS ==================
def load_documents():
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith((".docx", ".pdf", ".txt")):
            text = read_file(os.path.join(DATA_DIR, f))
            for chunk in chunk_text(text):
                if len(chunk.strip()) > 50 and is_asalari(chunk):
                    docs.append(chunk.strip())
    return docs

# ================== EMBEDDINGS ==================
def embed_texts(texts):
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        r = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        for d in r.data:
            vectors.append(d.embedding)
    return np.array(vectors).astype("float32")

# ================== BUILD / LOAD INDEX ==================
def build_index():
    docs = load_documents()
    if not docs:
        raise RuntimeError("❌ data/ ichida asalarichilik hujjatlari yo‘q")

    print(f"📄 Chunklar: {len(docs)}")
    vectors = embed_texts(docs)

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(docs, f)

    print("✅ FAISS index yaratildi")

def load_index():
    return faiss.read_index(INDEX_FILE), pickle.load(open(META_FILE, "rb"))

# ================== SEARCH ==================
def search_documents(question):
    index, texts = load_index()

    queries = [
        question,
        translate(question, "ru"),
        translate(question, "en")
    ]

    results = []

    for q in queries:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=[q]
        ).data[0].embedding

        D, I = index.search(np.array([emb]).astype("float32"), TOP_K)
        for i in I[0]:
            txt = texts[i]
            if is_asalari(txt):
                results.append(txt)

    return list(dict.fromkeys(results))[:TOP_K]

# ================== AI ANSWER ==================
def ai_answer(user_id, question):
    if not is_asalari(question):
        return "🐝 Bu bot faqat asalarichilik bo‘yicha savollarga javob beradi."

    lang = detect_lang(question)
    topic = detect_topic(question)
    contexts = search_documents(question)

    if not contexts:
        return "❌ Bu savol bo‘yicha hujjatlarda ma’lumot topilmadi."

    memory = user_memory.get(user_id, [])
    memory_text = "\n".join(memory[-2:]) if memory else ""

    prompt = f"""
You are a professional beekeeper assistant.

Topic: {topic}
Language: {lang}

Rules:
- ONLY beekeeping
- Practical and clear
- Step by step if possible

Documents:
{chr(10).join(contexts)}

Previous related questions:
{memory_text}

Question:
{question}
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert beekeeper."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

# ================== MEMORY ==================
def save_memory(user_id, question):
    mem = user_memory.get(user_id, [])
    mem.append(question)
    user_memory[user_id] = mem[-MAX_MEMORY:]

# ================== TELEGRAM ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐝 Asalarichilik AI bot\nUZ / RU / EN\nSavol bering."
    )

async def reindex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    await update.message.reply_text("♻️ Index yangilanmoqda...")
    build_index()
    await update.message.reply_text("✅ Tayyor")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    q = update.message.text.strip()

    await update.message.reply_text("⏳ O‘ylayapman...")

    ans = ai_answer(uid, q)
    save_memory(uid, q)

    for i in range(0, len(ans), 3500):
        await update.message.reply_text(ans[i:i+3500])

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_index()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reindex", reindex))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🐝 Asalarichilik AI Telegram bot ishga tushdi")
    app.run_polling()
