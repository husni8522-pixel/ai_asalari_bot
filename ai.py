from config import client
from utils import detect_lang, is_asalari, basic_chat
from indexer import search_docs

def ai_answer(uid, question):
    lang = detect_lang(question)

    basic = basic_chat(question)
    if basic:
        return basic[lang]

    if not is_asalari(question):
        return "🐝 Bot faqat asalarichilik uchun."

    context = "\n".join(search_docs(question))

    if not context:
        return "❌ Ma’lumot topilmadi."

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":"You are expert beekeeper."},
            {"role":"user","content":f"{context}\n\nSavol: {question}"}
        ],
        temperature=0.3
    )

    return r.choices[0].message.content.strip()
