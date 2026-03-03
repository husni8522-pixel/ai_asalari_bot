from config import client
from utils import basic_chat, t, is_asalari
from indexer import search_docs
from globals import user_memory, user_levels, user_languages


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

    # 1️⃣ Basic chat
    basic = basic_chat(uid, q)
    if basic:
        return basic

    # 2️⃣ Faqat asalarichilik
    if not is_asalari(q):
        return t(uid, "only_beekeeping")

    # 3️⃣ Til
    lang = user_languages.get(uid, "uz")

    lang_map = {
        "uz": "Uzbek",
        "ru": "Russian",
        "en": "English"
    }

    target_lang = lang_map.get(lang, "Uzbek")

    # 4️⃣ Memory
    if uid not in user_memory:
        user_memory[uid] = []

    user_memory[uid].append(q)

    # 5️⃣ FAISS
    ctx_list = search_docs(q)

    if not ctx_list:
        return t(uid, "only_beekeeping")

    ctx = "\n\n".join(ctx_list[:5])

    if len(ctx) > 4000:
        ctx = ctx[:4000]

    # 6️⃣ Level
    level = user_levels.get(uid, "beginner")

    if level == "beginner":
        max_tokens = 700
        temperature = 0.4

        role_description = "You are a friendly beekeeping teacher. Explain simply."

    elif level == "professional":
        max_tokens = 1200
        temperature = 0.3

        if is_disease_question(q):
            role_description = """
You are a senior veterinary beekeeping expert.
Provide structured disease explanation including:
- types
- symptoms
- causes
- treatment (active substances)
- prevention
"""
        else:
            role_description = "You are a professional beekeeping expert. Provide structured detailed answer."

    elif level == "ultra":
        max_tokens = 1500
        temperature = 0.2

        role_description = """
You are an ultra academic beekeeping expert.
Provide deep scientific explanation including biology, lifecycle,
pathogens, treatment protocols and resistance risks.
"""

    else:
        max_tokens = 800
        temperature = 0.3
        role_description = "Provide clear and accurate beekeeping answer."

    # 7️⃣ AI CALL
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
{role_description}

CRITICAL LANGUAGE RULE:
- Final answer MUST be in {target_lang}.
- Even if context is in another language, translate mentally.
- Never switch language.
"""
            },
            {
                "role": "user",
                "content": f"""
CONTEXT:
{ctx}

QUESTION:
{q}
"""
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer = response.choices[0].message.content.strip()

    # 8️⃣ Final strict translation (100% kafolat)
    if target_lang != "Uzbek":
        translation = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate this strictly into {target_lang}. Do not change meaning."
                },
                {
                    "role": "user",
                    "content": answer
                }
            ],
            temperature=0
        )

        answer = translation.choices[0].message.content.strip()

    return answer
