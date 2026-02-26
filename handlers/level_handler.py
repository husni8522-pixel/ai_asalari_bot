import time
import random
import pickle

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes

from globals import (
    user_test_state,
    user_test_cooldown,
    user_languages,
    user_levels,
    LEVELS_FILE
)

from test_data import TEST_QUESTIONS
from utils import t


# 🔥 Tugma yasovchi funksiya
def build_test_keyboard(options):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(opt, callback_data=f"test_{i}")]
        for i, opt in enumerate(options)
    ])


# 🔥 Professional testni boshlash
async def start_professional_test(update: Update, context: ContextTypes.DEFAULT_TYPE):

    uid = update.effective_user.id
    now = time.time()

    msg = update.callback_query.message if update.callback_query else update.message

    # 10 minut cheklov
    if uid in user_test_cooldown:
        last = user_test_cooldown[uid]
        if now - last < 600:
            remaining = int((600 - (now - last)) / 60)
            await msg.reply_text(t(uid, "cooldown", time=remaining))
            return

    shuffled = random.sample(TEST_QUESTIONS, 5)

    user_test_state[uid] = {
        "step": 0,
        "correct": 0,
        "questions": shuffled
    }

    lang = user_languages.get(uid, "uz")
    first_q = shuffled[0][lang]

    await msg.reply_text(
        f"{t(uid, 'test_start')}\n\n📊 1/5\n\n{first_q['question']}",
        reply_markup=build_test_keyboard(first_q["options"])
    )


# 🔥 Test javoblarini qabul qilish
async def test_answer_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    query = update.callback_query
    await query.answer()

    uid = query.from_user.id

    if uid not in user_test_state:
        return

    state = user_test_state[uid]
    lang = user_languages.get(uid, "uz")

    step = state["step"]
    questions = state["questions"]

    selected_index = int(query.data.split("_")[1])
    correct_index = questions[step][lang]["correct"]

    if selected_index == correct_index:
        state["correct"] += 1

    state["step"] += 1

    # 🔥 TEST TUGADI
    if state["step"] >= len(questions):

        if state["correct"] >= 4:
            user_levels[uid] = "professional"
            mode_name = t(uid, "mode_professional")
            await query.edit_message_text(t(uid, "test_success"))
        else:
            user_levels[uid] = "beginner"
            mode_name = t(uid, "mode_beginner")
            await query.edit_message_text(t(uid, "test_fail"))

        # 🔥 Levelni saqlaymiz (Railway volume)
        pickle.dump(user_levels, open(LEVELS_FILE, "wb"))

        user_test_cooldown[uid] = time.time()
        del user_test_state[uid]

        # 🔥 Rejim haqida xabar
        await query.message.reply_text(
            t(uid, "mode_active", mode=mode_name)
        )

        return

    # 🔥 Keyingi savolni EDIT qilamiz
    next_q = questions[state["step"]][lang]

    await query.edit_message_text(
        f"📊 {state['step']+1}/5\n\n{next_q['question']}",
        reply_markup=build_test_keyboard(next_q["options"])
    )
