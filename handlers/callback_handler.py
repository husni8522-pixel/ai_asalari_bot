from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from globals import user_languages
from handlers.level_handler import start_professional_test
from globals import user_levels
from utils import t

def reset_btn():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄", callback_data="reset")]
    ])


# 🔥 RESET
async def reset_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    await update.callback_query.message.reply_text(
        "✅",
        reply_markup=reset_btn()
    )


# 🔥 TIL TANLASH
async def lang_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    lang = q.data.split("_")[1]
    user_languages[q.from_user.id] = lang

    uid = q.from_user.id

    texts = {
        "uz": "🐝 Assalomu alaykum!\n\nDarajangizni tanlang:",
        "ru": "🐝 Здравствуйте!\n\nВыберите уровень:",
        "en": "🐝 Hello!\n\nChoose your level:"
    }

    level_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(uid, "level_beginner"), callback_data="level_beginner")],
        [InlineKeyboardButton(t(uid, "level_professional"), callback_data="level_pro")]
    ])

    await q.message.reply_text(
        texts.get(lang, texts["uz"]),
        reply_markup=level_kb
    )


# 🔥 LEVEL CALLBACK
async def level_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "level_pro":
        await start_professional_test(update, context)

    if q.data == "level_beginner":
        uid = q.from_user.id
        user_levels[uid] = "beginner"

        await q.message.reply_text(
            t(uid, "level_activated_beginner")
        )