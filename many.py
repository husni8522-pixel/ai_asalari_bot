import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from langdetect import detect
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
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

if not BOT_TOKEN or not OPENAI_KEY:
    raise RuntimeError("❌ .env faylda Telegram token yoki OpenAI key mavjud emas")

client = OpenAI(api_key=OPENAI_KEY)

# ================== MEMORY ==================
user_memory = {}  # user_id -> savol (faqat bitta savol konteksti)

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

# ================== ASALARI SO'ZLARI ==================
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
    "honey harvesting","honey extraction","honey quality"
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

    _, I = index.search(np.array([emb]).astype("float32"), TOP_K)
    return [texts[i] for i in I[0]]

# ================== AI ANSWER ==================
def ai_answer(uid, q):
    lang = detect_lang(q)

    # basic chat
    basic = basic_chat(q)
    if basic:
        return basic[lang]

    # 🔒 Agar oldingi savol mavjud bo‘lsa yangi savolni bloklaymiz
    if uid in user_memory and user_memory[uid]:
        return {
            "uz": "⚠️ Iltimos, avval 🔄 *Boshqa savol* tugmasini bosing.",
            "ru": "⚠️ Пожалуйста, сначала нажмите 🔄 *Новый вопрос*.",
            "en": "⚠️ Please press 🔄 *New question* first."
        }[lang]

    # faqat asalarichilik
    if not is_asalari(q):
        return {
            "uz": "🐝 Bu bot faqat asalarichilik bo‘yicha.",
            "ru": "🐝 Бот только для пчеловодства.",
            "en": "🐝 This bot is for beekeeping only."
        }[lang]

    # faqat bitta savolni eslab qolamiz
    user_memory[uid] = [q]

    ctx = "\n".join(search_docs(q))
    if not ctx:
        return {
            "uz": "❌ Ma’lumot topilmadi.",
            "ru": "❌ Информация не найдена.",
            "en": "❌ No information found."
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

# ================== BUTTON ==================
def reset_button():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Boshqa savol", callback_data="reset")]
    ])

# ================== HANDLERS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐝 Asalarichilik AI botga xush kelibsiz!\nSavol berishingiz mumkin.",
        reply_markup=reset_button()
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    q = update.message.text.strip()

    ans = ai_answer(uid, q)
    await update.message.reply_text(ans, reply_markup=reset_button())

async def reset_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    uid = query.from_user.id

    user_memory.pop(uid, None)

    await query.answer()
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="✅ Context tozalandi. Endi yangi savol berishingiz mumkin.",
        reply_markup=reset_button()
    )

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_index()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(reset_callback, pattern="^reset$"))

    print("🐝 BOT ISHGA TUSHDI")
    app.run_polling()
