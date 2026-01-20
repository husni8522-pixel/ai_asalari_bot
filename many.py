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
from datetime import datetime

# ================== CONFIG ==================
DATA_DIR = "data"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"

CHUNK_SIZE = 1000
BATCH_SIZE = 32
TOP_K = 8
MAX_MEMORY = 5

# ================== LOAD ENV ==================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))

if not BOT_TOKEN or not OPENAI_KEY:
    raise RuntimeError("❌ .env da token yoki openai key yo‘q")

client = OpenAI(api_key=OPENAI_KEY)

# ================== MEMORY & STATS ==================
user_memory = {}
user_stats = set()
questions_log = []

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

# ================== BASIC CHAT ==================
def basic_chat(text):
    t = text.lower()

    owner_uz = "Mening hujayinim Husniddin Zaripov, u juda yaxshi inson."
    owner_ru = "Мой хозяин — Хусниддин Зарипов, он очень хороший человек."
    owner_en = "My owner is Husniddin Zaripov. He is a very good person."

    if any(w in t for w in ["salom", "assalomu", "hello", "hi", "привет", "здравствуйте"]):
        return {
            "uz": "Assalomu alaykum 😊 Savolingizni yozing.",
            "ru": "Здравствуйте 😊 Задайте ваш вопрос.",
            "en": "Hello 😊 Please ask your question."
        }

    if any(w in t for w in ["kimsan", "kim sen", "who are you", "кто ты"]):
        return {
            "uz": "Men asalarichilik bo‘yicha aqlli yordamchi botman 🐝",
            "ru": "Я умный бот-помощник по пчеловодству 🐝",
            "en": "I am an intelligent beekeeping assistant bot 🐝"
        }

    if any(w in t for w in ["kim yaratgan", "kim tuzgan", "kim ixtiro", "owner", "создал", "invented"]):
        return {
            "uz": owner_uz,
            "ru": owner_ru,
            "en": owner_en
        }

    if any(w in t for w in ["telefon", "номер", "phone", "raqaming"]):
        return {
            "uz": "📞 Telefon raqam: +998973850026",
            "ru": "📞 Номер телефона: +998973850026",
            "en": "📞 Phone number: +998973850026"
        }

    return None

# ================== ASALARICHILIK ==================
ASALARI_WORDS = ["asal", "ari", "varroa", "qirolicha", "bee", "honey", "пчела", "мёд"]

def is_asalari(text):
    return any(w in text.lower() for w in ASALARI_WORDS)

# ================== FILE READ ==================
def read_file(path):
    if path.endswith(".docx"):
        return "\n".join(p.text for p in Document(path).paragraphs)
    if path.endswith(".pdf"):
        return "\n".join(p.extract_text() for p in PdfReader(path).pages if p.extract_text())
    if path.endswith(".txt"):
        return open(path, encoding="utf-8", errors="ignore").read()
    return ""

def chunk_text(text):
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

# ================== INDEX ==================
def build_index():
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith((".pdf", ".docx", ".txt")):
            text = read_file(os.path.join(DATA_DIR, f))
            for c in chunk_text(text):
                if len(c) > 50 and is_asalari(c):
                    docs.append(c)

    vectors = []
    for i in range(0, len(docs), BATCH_SIZE):
        r = client.embeddings.create(
            model="text-embedding-3-small",
            input=docs[i:i+BATCH_SIZE]
        )
        vectors.extend([d.embedding for d in r.data])

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, INDEX_FILE)
    pickle.dump(docs, open(META_FILE, "wb"))

def search_docs(q):
    index = faiss.read_index(INDEX_FILE)
    texts = pickle.load(open(META_FILE, "rb"))

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[q]
    ).data[0].embedding

    D, I = index.search(np.array([emb]).astype("float32"), TOP_K)
    return [texts[i] for i in I[0]]

# ================== AI ANSWER ==================
def ai_answer(uid, q):
    lang = detect_lang(q)

    basic = basic_chat(q)
    if basic:
        return basic[lang]

    if not is_asalari(q):
        return {
            "uz": "🐝 Bu bot asalarichilik uchun mo‘ljallangan.",
            "ru": "🐝 Этот бот предназначен для пчеловодства.",
            "en": "🐝 This bot is for beekeeping only."
        }[lang]

    ctx = "\n".join(search_docs(q))

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert beekeeper."},
            {"role": "user", "content": f"{ctx}\n\nSavol: {q}"}
        ],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

# ================== TELEGRAM ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stats.add(update.effective_user.id)
    await update.message.reply_text("🐝 Asalarichilik AI botga xush kelibsiz!")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    await update.message.reply_text(
        f"📊 Foydalanuvchilar: {len(user_stats)}\n"
        f"📩 Savollar: {len(questions_log)}"
    )

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    q = update.message.text
    user_stats.add(uid)

    ans = ai_answer(uid, q)
    questions_log.append(q)

    # ADMIN LOG
    await context.bot.send_message(
        ADMIN_ID,
        f"👤 USER: {uid}\n🕒 {datetime.now()}\n❓ {q}\n✅ {ans}"
    )

    await update.message.reply_text(ans)

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_index()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🐝 BOT ISHGA TUSHDI")
    app.run_polling()
