from config import client
from utils import basic_chat, t
from indexer import search_docs
from globals import user_memory, user_levels
from globals import current_ad

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

    ctx = "\n\n".join(ctx_list[:2])

    # 🔥 Contextni cheklaymiz (GPT haddan oshmasin)
    if len(ctx) > 1500:
        ctx = ctx[:1500]

    # 4️⃣ USER LEVEL
    level = user_levels.get(uid, "beginner")

    # =========================
    # 🌱 BEGINNER
    # =========================
    if level == "beginner":

        max_tokens = 400
        temperature = 0.4

        system_prompt = """
You are a friendly beekeeping teacher.

Rules:
- Always answer in the same language as the user.
- Explain in very simple words.
- Avoid scientific terminology.
- Keep answer short and clear.
- Use examples if helpful.
- Make it easy for new beekeepers to understand.
- Do not write long paragraphs.
If the answer is not clearly supported by the provided context,
say that the information is not available in the knowledge base.
Do not invent information.
"""

    # =========================
    # 🧠 PROFESSIONAL
    # =========================
    elif level == "professional":

        max_tokens = 900
        temperature = 0.3

        system_prompt = """
You are a professional beekeeping expert.

Important:
- If the question mentions 'kana', 'mite', 'varroa', treat it strictly as a parasite topic.
- Do NOT confuse bee species with parasites.
- If the question is about mites, answer only about parasitic mites affecting bees.

Rules:
- Always answer in the same language as the user.
- Be structured and precise.
If the answer is not clearly supported by the provided context,
say that the information is not available in the knowledge base.
Do not invent information.
"""

    # =========================
    # 🔬 ULTRA EXPERT (AKADEMIK)
    # =========================
    elif level == "ultra":

        max_tokens = 1100
        temperature = 0.2

        system_prompt = """
You are a professional beekeeping expert.

Important:
- If the question mentions 'kana', 'mite', 'varroa', treat it strictly as a parasite topic.
- Do NOT confuse bee species with parasites.
- If the question is about mites, answer only about parasitic mites affecting bees.

Rules:
- Always answer in the same language as the user.
- Be structured and precise.
If the answer is not clearly supported by the provided context,
say that the information is not available in the knowledge base.
Do not invent information.
"""

    else:
        max_tokens = 600
        temperature = 0.3
        system_prompt = "Answer clearly."

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

answer = r.choices[0].message.content.strip()

# 📣 Reklama qo‘shish
from globals import current_ad
if current_ad:
    answer = f"{answer}\n\n━━━━━━━━━━\n📣 {current_ad}"

return answer
