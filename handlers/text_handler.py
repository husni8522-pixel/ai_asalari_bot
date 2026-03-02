from telegram import Update
from telegram.ext import ContextTypes

from ai_engine import ai_answer
from image_ai import find_images_for_question
from config import ADMIN_ID
from utils import t

from globals import (
    ads,
    chat_log,
    user_stats,
    questions_log,
    STATS_FILE,
    user_test_state,
    user_levels,
    user_languages,
    user_test_cooldown
)

import pickle
import random
import time
from datetime import datetime


# 🔥 Uzun xabarni bo‘lib yuborish
async def send_long_message(message, text):
    MAX_LEN = 4000
    for i in range(0, len(text), MAX_LEN):
        await message.reply_text(text[i:i+MAX_LEN])


async def text_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):

    uid = u.effective_user.id
    txt = u.message.text

    # ====================================================
    # 🧠 PROFESSIONAL TEST LOGIKA (eski text test uchun)
    # ====================================================
    if uid in user_test_state:

        state = user_test_state[uid]
        lang = user_languages.get(uid, "uz")
        step = state["step"]
        questions = state["questions"]

        answer = txt.lower()
        keywords = questions[step][lang].get("keywords", [])

        if any(k in answer for k in keywords):
            state["correct"] += 1

        state["step"] += 1

        # Keyingi savol
        if state["step"] < len(questions):
            await u.message.reply_text(
                questions[state["step"]][lang]["question"]
            )
            return

        # Test tugadi
        if state["correct"] >= 4:
            user_levels[uid] = "professional"
            await u.message.reply_text(t(uid, "test_success"))
        else:
            user_levels[uid] = "beginner"
            await u.message.reply_text(t(uid, "test_fail"))

        user_test_cooldown[uid] = time.time()
        del user_test_state[uid]
        return

    # ====================================================
    # 🤖 AI JAVOB
    # ====================================================
    ans = ai_answer(uid, txt)

    # ====================================================
    # 📊 STATISTIKA
    # ====================================================
    try:
        user_stats.add(uid)
        questions_log.append(txt)

        with open(STATS_FILE, "wb") as f:
            pickle.dump({
                "users": user_stats,
                "questions": questions_log
            }, f)
    except:
        pass

    # ====================================================
    # 🖼 RASM QIDIRISH
    # ====================================================
    images = find_images_for_question(txt)

    for img in images:
        try:
            with open(img, "rb") as photo:
                await u.message.reply_photo(photo=photo)
        except:
            pass

    # ====================================================
    # 💬 AI JAVOB YUBORISH
    # ====================================================
    await send_long_message(u.message, ans)

    # ====================================================
    # 📣 REKLAMA (ESKI ADS LIST TIZIMI)
    # ====================================================
    # 🔥 REKLAMA (ads.pkl list tizimi)
    if isinstance(ads, list) and ads:
        ad_text = random.choice(ads)
        await u.message.reply_text(f"📣 Tavsiya qilamiz!\n\n{ad_text}")
    # ====================================================
    # 🛠 ADMIN LOG
    # ====================================================
    # 🔥 ADMIN LOG (Savol + Javob)
if ADMIN_ID:

    chat_title = chat_log.get(u.effective_chat.id, {}).get("title", "Private chat")
    chat_type = chat_log.get(u.effective_chat.id, {}).get("type", u.effective_chat.type)

    # Javobni juda uzun bo‘lsa qisqartiramiz
    short_answer = ans
    if len(short_answer) > 1500:
        short_answer = short_answer[:1500] + "\n\n... (javob qisqartirildi)"

    msg = (
        f"👤 USER ID: {uid}\n"
        f"🕒 {datetime.now()}\n"
        f"💬 Chat: {chat_title} ({chat_type})\n\n"
        f"❓ SAVOL:\n{txt}\n\n"
        f"🤖 JAVOB:\n{short_answer}"
    )

    await c.bot.send_message(chat_id=ADMIN_ID, text=msg)
        except:
            pass
