import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from langdetect import detect
from telegram import Update, File
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
    raise RuntimeError("❌ .env da token yoki OpenAI key yo‘q")

client = OpenAI(api_key=OPENAI_KEY)

# ================== MEMORY & STATS ==================
user_memory = {}  # user_id -> savollar
user_stats = set()  # foydalanuvchilar
questions_log = []  # savollar logi

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

    # Salomlashish
    if any(w in t for w in ["salom", "assalomu", "hello", "hi", "привет", "здравствуйте"]):
        return {
            "uz": "Assalomu alaykum 😊 Savolingizni yozing.",
            "ru": "Здравствуйте 😊 Задайте ваш вопрос.",
            "en": "Hello 😊 Please ask your question."
        }

    # Kim ekanligi
    if any(w in t for w in ["kimsan", "kim sen", "who are you", "кто ты"]):
        return {
            "uz": "Men asalarichilik bo‘yicha aqlli yordamchi botman 🐝",
            "ru": "Я умный бот-помощник по пчеловодству 🐝",
            "en": "I am an intelligent beekeeping assistant bot 🐝"
        }

    # Kim yaratgan
    if any(w in t for w in ["kim yaratgan", "kim tuzgan", "kim ixtiro", "owner", "создал", "invented"]):
        return {
            "uz": owner_uz,
            "ru": owner_ru,
            "en": owner_en
        }

    # Telefon raqam
    if any(w in t for w in ["telefon", "номер", "phone", "raqaming"]):
        return {
            "uz": "📞 Telefon raqam: +998973850026",
            "ru": "📞 Номер телефона: +998973850026",
            "en": "📞 Phone number: +998973850026"
        }

    return None

# ================== ASALARICHILIK ==================
ASALARI_WORDS = [
    # ---------- ASOSIY ----------
"ari","arilar","asal","asalarichilik","asalarichi",
"ари","арилар","асал","асаларичилик","асаларичи",
"bee","bees","honey","beekeeping","beekeeper",
"пчела","пчёлы","мёд","пчеловодство","пчеловод",

# ---------- ARI TURLARI ----------
"qirolicha","ona ari","ishchi ari","erkak ari","ari oilasi",
"қиролича","она ари","ишчи ари","эркак ари","ари оиласи",
"queen bee","worker bee","drone bee","bee colony",
"матка","рабочая пчела","трутень","пчелиная семья",

# ---------- UYALAR ----------
"ari uyasi","ari uyalari","katta uya","kichik uya","ko‘p qavatli uya",
"dadan","langstroth","rut","nukleus","bo‘linma uya",
"ари уяси","катта уя","кичик уя","кўп қаватли уя",
"улей","многокорпусный улей","лежак","дадан",
"hive","beehive","langstroth hive","dadant hive","nucleus hive",

# ---------- UYA QISMLARI ----------
"ramka","ramkalar","katak","sota","panjara",
"asos","mumli asos","asali panjara",
"рамка","рамки","соты","вощина","разделительная решётка",
"frame","frames","honeycomb","wax foundation","queen excluder",

# ---------- JIHOZLAR ----------
"asalarichi kiyimi","niqob","qo‘lqop","tutatuvchi",
"asal ajratgich","asal ekstraktori","asal pichog‘i",
"асаларичи кийими","ниқоб","қўлқоп","тутатувчи",
"дымарь","медогонка","нож для распечатки",
"beekeeper suit","veil","gloves","smoker","honey extractor",

# ---------- MAHSULOTLAR ----------
"asal","mum","propolis","perga","gulchang","qirollik suti","ari zahri",
"асал","мум","прополис","перга","гулчанг","маточное молочко",
"honey","wax","propolis","bee bread","pollen","royal jelly",

# ---------- KASALLIKLAR ----------
"varroa","nosema","akarapidoz","amerikan chirishi","yevropa chirishi",
"virus","zamburug‘","ari kasalligi",
"варроа","нозема","акарапидоз","гнилец","вирус","грибок",
"varroa mite","nosema disease","american foulbrood","viral disease",

# ---------- DAVOLASH ----------
"davolash","profilaktika","dori","kimyoviy davolash","organik davolash",
"oksalat kislota","formik kislota","timol",
"даволаш","профилактика","дори","щавелевая кислота","тимол",
"treatment","prevention","medicine","oxalic acid","formic acid",

# ---------- OZIQALANTIRISH ----------
"oziqlantirish","shakar","sirop","kandi","bahorgi oziqlantirish",
"озиқлантириш","шакар","сироп","канди",
"feeding","sugar","syrup","candy",

# ---------- PARVARISH ----------
"qishlatish","yozlatish","parvarish","ventilyatsiya","izolyatsiya",
"қишлатиш","парвариш","вентиляция",
"wintering","care","ventilation",

# ---------- ISHLAB CHIQARISH ----------
"asal yig‘ish","asal olish","asal ajratish","asal sifati","filtrlash",
"асал йиғиш","асал олиш","асал сифати",
"honey harvesting","honey extraction","honey quality",

# Oziqlantirish va tayyorlash
    "oziqlantirish", "shakar", "kandi", "sirop", "siroplar", "bal siropi", "bal shakar", "shakarli yem",
    "ari oziqlantirish", "ari ovqat", "bal bilan oziqlantirish", "ozuqa", "kand tayyorlash", "asalar ovqati",
]

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
    print("♻️ Indeks qurilmoqda...")
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith((".pdf", ".docx", ".txt")):
            text = read_file(os.path.join(DATA_DIR, f))
            for c in chunk_text(text):
                if len(c.strip()) > 50 and is_asalari(c):
                    docs.append(c.strip())
    if not docs:
        print("❌ Data papkada asalarichilik hujjatlari topilmadi!")
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
    print("✅ Indeks tayyor")

def search_docs(q):
    if not os.path.exists(INDEX_FILE):
        return []
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
            "uz": "🐝 Bu bot faqat asalarichilik uchun mo‘ljallangan.",
            "ru": "🐝 Этот бот предназначен для пчеловодства.",
            "en": "🐝 This bot is for beekeeping only."
        }[lang]

    ctx = "\n".join(search_docs(q))
    if not ctx:
        return {
            "uz": "❌ Bu savol bo‘yicha data papkada ma’lumot topilmadi.",
            "ru": "❌ По этому вопросу информация в папке data не найдена.",
            "en": "❌ No information found in data folder for this question."
        }[lang]

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert beekeeper."},
            {"role": "user", "content": f"{ctx}\n\nSavol: {q}"}
        ],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

# ================== TELEGRAM HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_stats.add(update.effective_user.id)
    await update.message.reply_text(
        "🐝 Asalarichilik AI botga xush kelibsiz!\nSavol berishingiz mumkin."
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    await update.message.reply_text(
        f"📊 Foydalanuvchilar: {len(user_stats)}\n"
        f"📩 Savollar: {len(questions_log)}"
    )

async def reindex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    await update.message.reply_text("♻️ Index yangilanmoqda...")
    build_index()
    await update.message.reply_text("✅ Index tayyor")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    q = update.message.text.strip()
    user_stats.add(uid)
    questions_log.append(q)

    ans = ai_answer(uid, q)

    # ADMIN LOG
    await context.bot.send_message(
        ADMIN_ID,
        f"👤 USER: {uid}\n🕒 {datetime.now()}\n❓ {q}\n✅ {ans}"
    )
    await update.message.reply_text(ans)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_stats.add(uid)
    photo = update.message.photo[-1]
    file: File = await photo.get_file()
    path = os.path.join("tmp", f"{photo.file_id}.jpg")
    os.makedirs("tmp", exist_ok=True)
    await file.download_to_drive(path)
    await update.message.reply_text("📷 Rasm qabul qilindi, tahlil qilinmoqda...")

    # AI javob (misol uchun rasmni tavsiflash)
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert beekeeper."},
            {"role": "user", "content": f"Bu rasmni tavsifla va agar kasallik bo'lsa qanday davo qilishni ayt:\n{path}"}
        ],
        temperature=0.3
    )
    ans = r.choices[0].message.content.strip()
    await context.bot.send_message(ADMIN_ID,
        f"👤 USER: {uid} (rasm)\n🕒 {datetime.now()}\n✅ {ans}"
    )
    await update.message.reply_text(ans)

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_index()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("reindex", reindex))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("🐝 BOT ISHGA TUSHDI")
    app.run_polling()
