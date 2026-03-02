from config import client
from utils import basic_chat, t
from indexer import search_docs
from globals import user_memory, user_levels

def is_disease_question(q):
    keywords = [
        "kana", "kanasi", "varroa", "kasallik",
        "disease", "mite", "treatment",
        "davolash", "dorilar", "zararkunanda"
    ]
    q_lower = q.lower()
    return any(k in q_lower for k in keywords)
    
def ai_answer(uid, q):

    # 1️⃣ Basic chat
    basic = basic_chat(uid, q)
    if basic:
        return basic

    # 2️⃣ User memory
    if uid not in user_memory:
        user_memory[uid] = []

    user_memory[uid].append(q)

    # 3️⃣ FAISS context
    ctx_list = search_docs(q)

    if not ctx_list:
        return t(uid, "only_beekeeping")

    # Eng yaqin 2 ta kontekst
    ctx = "\n\n".join(ctx_list[:5])

    # Context uzunligini cheklash
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

        system_prompt = """
You are a friendly beekeeping teacher.

Rules:
- Always answer in the same language as the user.
- Explain in very simple words.
- Avoid scientific terminology.
- Keep answer short and clear.
- Use examples if helpful.
- Do not invent information.
"""

    # =========================
    # 🧠 PROFESSIONAL
    # =========================
    elif level == "professional":

        max_tokens = 1200
        temperature = 0.3

    if is_disease_question(q):

        system_prompt = """
You are a senior veterinary beekeeping expert.

Rules:
- Always answer in the same language as the user.
- Be highly professional.
- Use structured sections.
- Include practical treatment details.
- Mention real medications and active substances.
- Be concise but complete.

Structure your answer EXACTLY like this:

🦠 Kasallik yoki zararkunanda nomi

📌 Turlari:
- (list types)

🔍 Belgilari:
- (symptoms)

⚠ Sabablari:
- (causes)

💊 Davolash:
- (treatment methods)
- (chemical treatments with active substances)
- (organic options if available)

🛡 Oldini olish:
- (prevention steps)

📌 Amaliy tavsiya:
- (short practical advice)
"""
    else:

        system_prompt = """
You are a professional beekeeping expert.

Rules:
- Always answer in the same language as the user.
- Provide structured, professional explanation.
- Include biological and technical details.
- Add practical recommendations.

Structure:

📘 Tushuntirish
🔬 Ilmiy izoh
⚙ Amaliy tavsiyalar
"""

    # =========================
    # 🔬 ULTRA
    # =========================
    elif level == "ultra":
        max_tokens = 2000
        temperature = 0.2

        system_prompt = """
You are an ultra expert academic beekeeping specialist.

Rules:
- Always answer in the same language as the user.
- Provide deep scientific explanation.
- Use biological and technical terminology.
- Be highly structured.
- Do not invent information.
"""

    else:
        max_tokens = 600
        temperature = 0.3
        system_prompt = "Answer clearly and accurately."

    # 5️⃣ AI CALL
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{ctx}\n\nQuestion: {q}"}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return r.choices[0].message.content.strip()


