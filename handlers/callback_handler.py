from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from globals import user_languages, user_levels
from handlers.level_handler import start_professional_test
from utils import t


# 🔥 RESET
async def reset_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("✅")


# 🔥 TIL TANLASH
async def lang_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    lang = query.data.split("_")[1]
    uid = query.from_user.id

    user_languages[uid] = lang

    texts = {
        "uz": "🐝 Assalomu alaykum!\n\nDarajangizni tanlang:",
        "ru": "🐝 Здравствуйте!\n\nВыберите уровень:",
        "en": "🐝 Hello!\n\nChoose your level:"
    }

    level_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(uid, "level_beginner"), callback_data="level_beginner")],
        [InlineKeyboardButton(t(uid, "level_professional"), callback_data="level_pro")]
    ])

    await query.edit_message_text(
        texts.get(lang, texts["uz"]),
        reply_markup=level_kb
    )


# 🔥 LEVEL CALLBACK
async def level_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    uid = query.from_user.id

    if query.data == "level_pro":
        await start_professional_test(update, context)
        return

    if query.data == "level_beginner":
        user_levels[uid] = "beginner"

        await query.edit_message_text(
            t(uid, "level_activated_beginner")
        )
        return
