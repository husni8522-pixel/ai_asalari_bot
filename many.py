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
    if any(w in t for w in ["salom", "assalomu", "hello", "hi", "привет"]):
        return {
            "uz": "Assalomu alaykum 😊 Savolingizni yozing.",
            "ru": "Здравствуйте 😊 Задайте вопрос.",
            "en": "Hello 😊 Ask your question."
        }
    return None

# ================== ASALARI ==================
ASALARI_WORDS = [
    "ari","arilar","asal","asalarichilik","asalarichi",
    "ари","арилар","асал","асаларичилик","асаларичи",
    "bee","bees","honey","beekeeping","beekeeper",
    "пчела","пчёлы","мёд","пчеловодство","пчеловод",
    "qirolicha","ona ari","ishchi ari","erkak ari","ari oilasi",
    "қиролича","она ари","ишчи ари","эркак ари","ари оиласи",
    "queen bee","worker bee","drone bee","bee colony",
    "матка","рабочая пчела","трутень","пчелиная семья",
    "ari uyasi","ari uyalari","katta uya","kichik uya","ko‘p qavatli uya",
    "dadan","langstroth","rut","nukleus","bo‘linma uya",
    "ари уяси","катта уя","кичик уя","кўп қаватли уя",
    "улей","многокорпусный улей","лежак","дадан",
    "hive","beehive","langstroth hive","dadant hive","nucleus hive",
    "ramka","ramkalar","katak","sota","panjara",
    "asos","mumli asos","asali panjara",
    "рамка","рамки","соты","вощина","разделительная решётка",
    "frame","frames","honeycomb","wax foundation","queen excluder",
    "asalarichi kiyimi","niqob","qo‘lqop","tutatuvchi",
    "asal ajratgich","asal ekstraktori","asal pichog‘i",
    "асаларичи кийими","ниқоб","қўлқоп","тутатувчи",
    "дымарь","медогонка","нож для распечатки",
    "beekeeper suit","veil","gloves","smoker","honey extractor",
    "asal","mum","propolis","perga","gulchang","qirollik suti","ari zahri",
    "асал","мум","прополис","перга","гулчанг","маточное молочко",
    "honey","wax","propolis","bee bread","pollen","royal jelly",
    "varroa","nosema","akarapidoz","amerikan chirishi","yevropa chirishi",
    "virus","zamburug‘","ari kasalligi",
    "варроа","нозема","акарапидоз","гнилец","вирус","грибок",
    "varroa mite","nosema disease","american foulbrood","viral disease",
    "davolash","profilaktika","dori","kimyoviy davolash","organik davolash",
    "oksalat kislota","formik kislota","timol",
    "даволаш","профилактика","дори","щавелевая кислота","тимол",
    "treatment","prevention","medicine","oxalic acid","formic acid",
    "oziqlantirish","shakar","sirop","kandi","bahorgi oziqlantirish",
    "озиқлантириш","шакар","сироп","канди",
    "feeding","sugar","syrup","candy",
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
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith((".pdf", ".docx", ".txt")):
            text = read_file(os.path.join(DATA_DIR, f))
            for c in chunk_text(text):
                if len(c.strip()) > 50 and is_asalari(c):
                    docs.append(c.strip())
    if not docs:
        return
    vectors = []
    for i in range(0, len(docs), BATCH_SIZE):
        r = client.embeddings.create(model="text-embedding-3-small", input=docs[i:i+BATCH_SIZE])
        vectors.extend([d.embedding for d in r.data])
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, INDEX_FILE)
    pickle.dump(docs, open(META_FILE, "wb"))

def search_docs(q):
    if not os.path.exists(INDEX_FILE):
        return []
    index = faiss.read_index(INDEX_FILE)
    texts = pickle.load(open(META_FILE, "rb"))
    emb = client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding
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
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Yangi savol", callback_data="reset")]])

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
        "🐝 Asalarichilik AI botga xush kelibsiz!",
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

# ================== ADMIN ==================
async def reindex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Sizda bu komandani ishlatish huquqi yo‘q.")
        return
    await update.message.reply_text("♻️ Indeks yangilanmoqda...")
    build_index()
    await update.message.reply_text("✅ Indeks tayyor")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Sizda bu komandani ishlatish huquqi yo‘q.")
        return
    chats = "\n".join([f"{v['title']} ({v['type']})" for v in chat_log.values()])
    await update.message.reply_text(
        f"📊 Foydalanuvchilar: {len(user_stats)}\n"
        f"📩 Savollar: {len(questions_log)}\n"
        f"💬 Guruhlar/kanallar:\n{chats}"
    )

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_index()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reindex", reindex))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(reset_callback, pattern="^reset$"))

    print("🐝 BOT ISHGA TUSHDI")
    app.run_polling()