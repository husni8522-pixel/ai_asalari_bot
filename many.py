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
    # ===== UZBEKCHA =====
"ari","arilar","asal","asalarichilik","asalarichi","ari oilasi","qirolicha","ona ari","ishchi ari","erkak ari",
"matka","truten","ari uyasi","katta uya","kichik uya","ko‘p qavatli uya","bo‘linma uya","ramka","katak","sota","panjara",
"mumli asos","asali panjara","asal ajratgich","asal ekstraktori","asal pichog‘i","asalarichi kiyimi","niqob","qo‘lqop",
"tutatuvchi","dimar","medogonka","ari zahri","qirollik suti","perga","gulchang","propolis","mum","honeycomb",
"oziqlantirish","shakar","sirop","kandi","bahorgi oziqlantirish","kuzgi oziqlantirish","qandari","tog‘ ari","suvli ari",
"quyoshli ari","italyanari","karlik ari","kafkasari","rus ari","yevropeysari","karniyari","himalayari","afrikari",
"medonosari","yovvoyi ari","asl ari","o‘zbek ari","qora ari","shakarli ari","o‘rta yevropalik ari",
"davolash","profilaktika","dori","kimyoviy davolash","organik davolash","oksalat kislota","formik kislota","timol",
"kasalliklar","varroa","nosema","akarapidoz","amerikan chirishi","yevropa chirishi","virus","zamburug‘",
"ari kasalligi","jarayonlar","buzilishni oldini olish","samaradorlik","honey harvest","swarm prevention","feeding syrup",
"nectar collection","pollen collection","queen marking","brood inspection","colony management","hive inspection",
"queen cage","honey frame","brood frame","wax frame","foundation sheet","cappings","supers","brood box","honey super",
"apiary","beekeeper journal","inspection report","nectar flow","honey flow","protein supplement","bee genetics",
"bee space","uncapping fork","honey gate","hive tool","bee brush","bee feeder","swarm trap","swarm box","nectar trap",
"pollination","queen rearing","artificial insemination","colony splitting","winter preparation","spring preparation",
"feeding candy","feeding syrup","feeding pollen","feeding protein","wax foundation replacement","frame rotation",
"queen introduction","drone management","varroa treatment","nosema treatment","american foulbrood treatment",
"european foulbrood treatment","wax moth treatment","hive ventilation","temperature control","humidity control",
"smoker management","medogonka cleaning","extractor maintenance","bee suit maintenance","gloves cleaning","veil cleaning",
"bee health check","disease prevention","pollen analysis","honey analysis","royal jelly harvesting","bee venom collection",
"bee venom extraction","propagation","queen selection","swarm capture","swarm relocation","colony boosting",
"bee identification","apiary mapping","hive numbering","hive labelling","inspection schedule","feed schedule",
"winter feeding","summer feeding","autumn feeding","spring feeding","nectar monitoring","pollen monitoring",
"beekeeping records","colony performance","honey production","wax production","propolis production","perga storage",
"honey storage","wax storage","hive hygiene","apiary hygiene","hive spacing","apiary layout","swarm behavior",
"bee behavior","foraging behavior","colony development","brood development","queen development","drone development",
"hive maintenance","frame repair","foundation repair","honey extraction","wax rendering","beekeeping equipment","apiary security",
# ===== RUSCHA =====
"пчела","пчёлы","мёд","пчеловодство","пчеловод","пчелиная семья","матка","трутень","рабочая пчела","улий",
"многокорпусный улей","рамка","соты","вощина","разделительная решётка","медогонка","пчелиная одежда","маска",
"перчатки","тутатучий","дымарь","нож для распечатки","перга","гулчан","прополис","воск","маточное молочко",
"кормление","сахар","сироп","кормовая паста","весенняя подкормка","осенняя подкормка","дикая пчела",
"итальянская пчела","карликовая пчела","кавказская пчела","русская пчела","европейская пчела","карнийская пчела",
"гималайская пчела","африканская пчела","медоносная пчела","местная пчела","чёрная пчела","солнечная пчела","водная пчела",
"лечение","профилактика","лекарство","химическое лечение","органическое лечение","оксаловая кислота","формическая кислота","тимол",
"болезни","варроа","нозема","акарапидоз","американский гнилец","европейский гнилец","вирус","грибок","процесс","сбор мёда",
"предотвращение роения","сироп для кормления","сбор нектара","сбор пыльцы","отметка матки","инспекция расплода","управление семьей",
"осмотр улья","клетка для матки","рамка с медом","рамка с расплодом","рамка с вощиной","вощина","суперы","коробка с расплодом",
"супер с медом","пасека","журнал пчеловода","отчет об инспекции","поток нектара","поток меда","протеиновая добавка",
"генетика пчёл","пространство пчел","вилка для распечатки","ворота для меда","инструмент для улья","щетка для пчел","кормушка для пчёл",
"ловушка для роя","коробка для роя","ловушка для нектара","опыление","разведение маток","искусственное осеменение","деление семьи",
"подготовка к зиме","подготовка к весне","кормление сахаром","кормление сиропом","кормление пыльцой","кормление белком",
"замена вощины","поворот рамки","введение матки","управление трутнями","лечение варроа","лечение ноземы","лечение американского гнильца",
"лечение европейского гнильца","лечение вощинной моли","вентиляция улья","контроль температуры","контроль влажности","уход за дымарем",
"чистка медогонки","обслуживание экстрактора","уход за костюмом","чистка перчаток","чистка маски","проверка здоровья пчёл",
"профилактика заболеваний","анализ пыльцы","анализ мёда","сбор маточного молочка","сбор пчелиного яда","экстракция пчелиного яда",
"размножение","отбор маток","поймать рой","переселение роя","усиление семьи","идентификация пчёл","карта пасеки",
"нумерация ульев","маркировка ульев","график инспекции","график кормления","зимнее кормление","летнее кормление",
"осеннее кормление","весеннее кормление","мониторинг нектара","мониторинг пыльцы","записи пчеловодства","производительность семьи",
"производство мёда","производство воска","производство прополиса","хранение перги","хранение мёда","хранение воска",
"гигиена улья","гигиена пасеки","размещение ульев","планировка пасеки","поведение роя","поведение пчёл",
"поведение при сборе нектара","развитие семьи","развитие расплода","развитие матки","развитие трутней","обслуживание улья",
"ремонт рамки","ремонт вощины","сбор мёда","переработка воска","оборудование пасеки","безопасность пасеки",
# ===== ENGLISH =====
"bee","bees","honey","beekeeping","beekeeper","bee colony","queen bee","worker bee","drone bee","hive","beehive","nucleus hive",
"langstroth hive","frames","honeycomb","wax foundation","queen excluder","beekeeper suit","veil","gloves","smoker",
"honey extractor","propolis","royal jelly","bee bread","pollen","wax","feeding","sugar","syrup","candy","spring feeding",
"autumn feeding","candy feeding","drone bee","queen rearing","artificial insemination","colony splitting","winter prep",
"spring prep","nectar collection","pollen collection","swarm prevention","swarm capture","swarm relocation","colony boosting",
"bee identification","apiary mapping","hive numbering","hive labelling","inspection schedule","feed schedule","winter feeding",
"summer feeding","autumn feeding","spring feeding","nectar monitoring","pollen monitoring","beekeeping records","colony performance",
"honey production","wax production","propolis production","perga storage","honey storage","wax storage","hive hygiene","apiary hygiene",
"hive spacing","apiary layout","swarm behavior","bee behavior","foraging behavior","colony development","brood development",
"queen development","drone development","hive maintenance","frame repair","foundation repair","honey extraction","wax rendering",
"beekeeping equipment","apiary security","varroa treatment","nosema treatment","american foulbrood treatment","european foulbrood treatment",
"wax moth treatment","hive ventilation","temperature control","humidity control","smoker maintenance","medogonka cleaning",
"extractor maintenance","bee suit maintenance","gloves cleaning","veil cleaning","bee health check","disease prevention",
"pollen analysis","honey analysis","royal jelly harvesting","bee venom collection","bee venom extraction","propagation",
"queen selection","colony inspection","honey frame","brood frame","wax frame","foundation sheet","cappings","supers","brood box",
"honey super","queen marking","bee brush","bee feeder","swarm trap","swarm box","nectar trap","pollination"
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
