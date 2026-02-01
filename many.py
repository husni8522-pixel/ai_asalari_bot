import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from langdetect import detect
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
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

client = OpenAI(api_key=OPENAI_KEY)

# ================== MEMORY & LOG ==================
user_memory = {}      # user_id -> savollar
questions_log = []    # savollar logi
user_stats = set()    # user_id lar
chat_log = {}         # chat_id -> {"title": str, "type": str}

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
    
    # SALOM / HAYRLASHUV
    if any(w in t for w in ["salom", "assalomu", "hello", "hi", "привет"]):
        return {
            "uz": "Assalomu alaykum 😊 nima xizmat. savol bormi?",
            "ru": "Здравствуйте 😊 Задайте вопрос.",
            "en": "Hello 😊 Ask your question."
        }
    if any(w in t for w in ["xayr", "hayr", "goodbye", "bye", "пока", "до свидания"]):
        return {
            "uz": "Xayr! Sizni kutib qolamiz 😊",
            "ru": "До свидания! Будем рады вас видеть снова 😊",
            "en": "Goodbye! We hope to see you again 😊"
        }
    
    # RAHMAT / MINNATDORLIK
    if any(w in t for w in ["rahmat", "raxmat", "рахмат", "спасибо", "thank you"]):
        return {
            "uz": "Siz uchun hursandman. Arzimaydi 😊",
            "ru": "Рад помочь! Не за что 😊",
            "en": "I’m happy to help. You’re welcome 😊"
        }
    
    # BOTNING YARATUVCHISI / ALOQA
    if any(w in t for w in ["sani kim yaratgan", "seni kim yaratgan", "sani kim tuzgan", "seni kim tuzgan", "sani hujayining kim", 
    "seni hujayining kim", "sen kim", "sani kim", "kim san", "kim sen", "kim tuzgan"]):
        return {
            "uz": "Men akajonim Husniddin Zaripov tomonidan yaratilgan botman. Akamni duolarizda eslab qo‘ying @zhn8522😊",
            "ru": "Мен создан ботом Хусниддин Зарипов. Помните моего брата в ваших молитвах @zhn8522 😊",
            "en": "I am a bot created by Husniddin Zaripov. Keep my brother in your prayers @zhn8522 😊"
        }
    if any(w in t for w in ["qanday aloqaga chiqamiz", "aloqa", "contact", "how to contact"]):
        return {
            "uz": "Aloqa uchun: +998973850026 📞",
            "ru": "Связь: +998973850026 📞",
            "en": "Contact: +998973850026 📞"
        }
    
    return None

# ================== ASALARI ==================
ASALARI_WORDS = [
    "ari","asalari ich ketishi","asalarim","qishki ozuqa","arilar","asal","asalarichilik",
    "asalarichi","ari oilasi","qirolicha","ona ari","ishchi ari","erkak ari","qandi",
    "kandi","nuklius","asalarilarim",
]

def is_asalari(text):
    return any(w in text.lower() for w in ASALARI_WORDS)

# ================== FILES ==================
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
    print("♻️ INDEX YARATILYAPTI...")
    docs = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for f in os.listdir(DATA_DIR):
        if f.endswith((".pdf", ".docx", ".txt")):
            text = read_file(os.path.join(DATA_DIR, f))
            for c in chunk_text(text):
                if len(c.strip()) > 50 and is_asalari(c):
                    docs.append(c.strip())

    if not docs:
        print("❌ DATA papkada mos hujjat yo‘q")
        return

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

    print("✅ INDEX TAYYOR")

def index_invalid():
    if not os.path.exists(INDEX_FILE):
        return True
    if not os.path.exists(META_FILE):
        return True
    if os.path.getsize(INDEX_FILE) < 1000:
        return True
    if os.path.getsize(META_FILE) < 50:
        return True
    return False

def search_docs(q):
    index = faiss.read_index(INDEX_FILE)
    texts = pickle.load(open(META_FILE, "rb"))
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[q]
    ).data[0].embedding

    _, I = index.search(np.array([emb]).astype("float32"), TOP_K)
    return [texts[i] for i in I[0]]

# ================== AI ANSWER ==================
def ai_answer(uid, q):
    lang = detect_lang(q)
    basic = basic_chat(q)
    if basic:
        return basic[lang]

    if uid not in user_memory:
        user_memory[uid] = []

    if not is_asalari(q):
        return {
            "uz": "🐝 Bu bot faqat asalarichilik uchun.",
            "ru": "🐝 Бот только для пчеловодства.",
            "en": "🐝 This bot is for beekeeping only."
        }[lang]

    user_memory[uid].append(q)

    ctx = "\n".join(search_docs(q))
    if not ctx:
        return {
            "uz": "❌ Ma’lumot topilmadi.",
            "ru": "❌ Информация не найдена.",
            "en": "❌ No information found."
        }[lang]

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": "You are an expert beekeeper."},
                  {"role": "user", "content": f"{ctx}\n\nSavol: {q}"}],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

# ================== BUTTON ==================
def reset_button():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Yangi | Новый | New", callback_data="reset")]])

# ================== LOG CHAT ==================
async def log_chat(update: Update):
    chat = update.effective_chat
    user_stats.add(update.effective_user.id)
    if chat.id not in chat_log:
        chat_log[chat.id] = {
            "title": chat.title or f"{update.effective_user.first_name}",
            "type": chat.type
        }

# ================== HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await log_chat(update)
    await update.message.reply_text(
        "🐝 Asalarichilik AI botga xush kelibsiz! Marhamat savol bering",
        reply_markup=reset_button()
    )
 
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await log_chat(update)
    uid = update.effective_user.id
    q = update.message.text.strip()
    questions_log.append(q)
    ans = ai_answer(uid, q)

    await update.message.reply_text(ans, reply_markup=reset_button())

    if ADMIN_ID:
        await context.bot.send_message(
            ADMIN_ID,
            f"👤 USER ID: {uid}\n🕒 {datetime.now()}\n❓ Savol: {q}\n✅ Javob: {ans}\n"
            f"💬 Chat: {chat_log[update.effective_chat.id]['title']} ({chat_log[update.effective_chat.id]['type']})"
        )

async def reset_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    uid = query.from_user.id
    user_memory.pop(uid, None)
    await query.answer()
    await query.message.reply_text(
        "✅ Context tozalandi. Yangi savol berishingiz mumkin.",
        reply_markup=reset_button()
    )

# ================== ADMIN PANEL ==================
ADMIN_CHOOSE, ADMIN_UPLOAD, ADMIN_DELETE = range(3)

def admin_keyboard():
    buttons = [
        [InlineKeyboardButton("📂 Fayl yuklash", callback_data="upload")],
        [InlineKeyboardButton("❌ Fayl uchirish", callback_data="delete")],
        [InlineKeyboardButton("📊 Foydalanuvchilar statistikasi", callback_data="stats")],
        [InlineKeyboardButton("🔄 Indeksni yangilash", callback_data="reindex")],
        [InlineKeyboardButton("⬅️ Chiqish", callback_data="exit")]
    ]
    return InlineKeyboardMarkup(buttons)

async def admin_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Sizda admin huquqi yo‘q")
        return ConversationHandler.END
    await update.message.reply_text("🔑 Admin panelga xush kelibsiz", reply_markup=admin_keyboard())
    return ADMIN_CHOOSE

async def admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "upload":
        await query.message.reply_text("📂 Faylni yuboring (.txt, .pdf, .docx)")
        return ADMIN_UPLOAD
    elif data == "delete":
        files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
        if not files:
            await query.message.reply_text("❌ Hech qanday fayl mavjud emas")
            return ADMIN_CHOOSE
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(f, callback_data=f"del_{f}")] for f in files])
        await query.message.reply_text("❌ Qaysi faylni uchirmoqchisiz?", reply_markup=kb)
        return ADMIN_DELETE
    elif data.startswith("del_"):
        fname = data[4:]
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            await query.message.reply_text(f"✅ {fname} fayl uchirildi")
            build_index()  # indeksni yangilash
        return ADMIN_CHOOSE
    elif data == "stats":
        chats = "\n".join([f"{v['title']} ({v['type']})" for v in chat_log.values()])
        await query.message.reply_text(
            f"📊 Foydalanuvchilar: {len(user_stats)}\n"
            f"📩 Savollar: {len(questions_log)}\n"
            f"💬 Guruhlar/kanallar:\n{chats}"
        )
        return ADMIN_CHOOSE
    elif data == "reindex":
        await query.message.reply_text("♻️ Indeks yangilanmoqda...")
        build_index()
        await query.message.reply_text("✅ Indeks tayyor")
        return ADMIN_CHOOSE
    elif data == "exit":
        await query.message.reply_text("⬅️ Admin paneldan chiqdingiz")
        return ConversationHandler.END
    return ADMIN_CHOOSE

async def admin_upload_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        await update.message.reply_text("❌ Fayl yuboring")
        return ADMIN_UPLOAD

    if not doc.file_name.endswith((".txt", ".pdf", ".docx")):
        await update.message.reply_text("❌ Faqat .txt, .pdf, .docx fayllar qabul qilinadi")
        return ADMIN_UPLOAD

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    path = os.path.join(DATA_DIR, doc.file_name)
    await doc.get_file().download_to_drive(path)
    await update.message.reply_text(f"✅ {doc.file_name} fayl yuklandi")

    build_index()  # fayl yuklanganda indeksni yangilash
    return ADMIN_CHOOSE

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_index()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Foydalanuvchi
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(reset_callback, pattern="^reset$"))

    # Admin panel
    admin_conv = ConversationHandler(
        entry_points=[CommandHandler("admin", admin_start)],
        states={
            ADMIN_CHOOSE: [CallbackQueryHandler(admin_callback)],
            ADMIN_UPLOAD: [MessageHandler(filters.Document.ALL, admin_upload_file)],
            ADMIN_DELETE: [CallbackQueryHandler(admin_callback)]
        },
        fallbacks=[CommandHandler("admin", admin_start)],
        allow_reentry=True
    )
    app.add_handler(admin_conv)

    print("🐝 BOT ISHGA TUSHDI")
    app.run_polling()
