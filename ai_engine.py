from config import client
from utils import basic_chat, t
from indexer import search_docs
from globals import user_memory, user_levels
from globals import user_languages
from utils import is_asalari

# 🔍 Kasallik savolini aniqlash
def is_disease_question(q):
    keywords = [
        "kana", "kanasi", "varroa", "kasallik",
        "disease", "mite", "treatment",
        "davolash", "dorilar", "zararkunanda"
    ]
    q_lower = q.lower()
    return any(k in q_lower for k in keywords)


def ai_answer(uid, q):

    lang = user_languages.get(uid, "uz")

    lang_map = {
        "uz": "Uzbek",
        "ru": "Russian",
        "en": "English"
    }

    target_lang = lang_map.get(lang, "Uzbek")
    
    # 1️⃣ Basic chat
    basic = basic_chat(uid, q)
    if basic:
        return basic
        
# 🔐 1️⃣ KEYWORD FILTER
    if not is_asalari(q):
        return t(uid, "only_beekeeping")
    
    # 2️⃣ User memory
    if uid not in user_memory:
        user_memory[uid] = []

    user_memory[uid].append(q)

    # 3️⃣ FAISS context
    ctx_list = search_docs(q)

    # 🔐 1️⃣ KEYWORD FILTER
    if not is_asalari(q):
        return t(uid, "only_beekeeping")
    
    if not ctx_list:
        return t(uid, "only_beekeeping")

    ctx = "\n\n".join(ctx_list[:5])

    if len(ctx) > 4000:
        ctx = ctx[:4000]

    # 4️⃣ USER LEVEL
    level = user_levels.get(uid, "beginner")

    # =========================
    # 🌱 BEGINNER
    # =========================
    if level == "beginner":

        max_tokens = 800
        temperature = 0.4

        system_prompt = f"""
You are a friendly beekeeping teacher.

Language Rule:
- You MUST answer ONLY in {target_lang}.
- Even if the user writes in another language, you MUST answer in {target_lang}.
- Do NOT switch language under any circumstance.

Rules:
- Provide structured professional explanation.
- Include biological and technical details.
- Add practical recommendations.
"""
    # =========================
    # 🧠 PROFESSIONAL
    # =========================
    elif level == "professional":

        max_tokens = 1200
        temperature = 0.3

        # 🔥 Kasallik bo‘lsa alohida professional format
        if is_disease_question(q):

            system_prompt = f"""
You are a senior veterinary beekeeping expert.

Rules:
- Always answer in the same language as the user.
- Be highly professional.
- Mention real medications and active substances.
- Be concise but complete.

Structure:

🦠 Kasallik yoki zararkunanda nomi

📌 Turlari:
🔍 Belgilari:
⚠ Sabablari:
💊 Davolash:
🛡 Oldini olish:
📌 Amaliy tavsiya:
"""

        else:

            system_prompt = f"""
You are a professional beekeeping expert.

Language Rule:
- You MUST answer ONLY in {target_lang}.
- Even if the user writes in another language, you MUST answer in {target_lang}.
- Do NOT switch language under any circumstance.

Rules:
- Provide structured professional explanation.
- Include biological and technical details.
- Add practical recommendations.
"""

    # =========================
    # 🔬 ULTRA
    # =========================
    elif level == "ultra":

        max_tokens = 1500
        temperature = 0.2

        system_prompt = f"""
You are an ultra expert academic beekeeping specialist.

Language Rule:
- You MUST answer ONLY in {target_lang}.
- Even if the user writes in another language, you MUST answer in {target_lang}.
- Do NOT switch language under any circumstance.

Rules:
- Provide structured professional explanation.
- Include biological and technical details.
- Add practical recommendations.
- Always answer in the same language as the user.
- Provide deep scientific explanation.
- Include pathogen biology, lifecycle and treatment protocols.
- Mention active substances and resistance risks.
- Be highly structured.
- Do not invent information.
"""

    # =========================
    # 🔄 DEFAULT
    # =========================
    else:

        max_tokens = 800
        temperature = 0.3
        system_prompt = "Answer clearly and accurately."

    # 5️⃣ AI CALL
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Answer strictly in {target_lang} language.\n\n{ctx}\n\nQuestion:\n{q}"
            }
    ],
    temperature=temperature,
    max_tokens=max_tokens
)

    return r.choices[0].message.content.strip()





