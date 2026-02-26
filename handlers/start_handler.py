from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from globals import chat_log


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # Chat log saqlash
    chat_log[update.effective_chat.id] = {
        "title": update.effective_chat.title or "Private chat",
        "type": update.effective_chat.type
    }

    # 🔥 Til tanlash tugmalari
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🇺🇿 O‘zbekcha", callback_data="lang_uz")],
        [InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru")],
        [InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")]
    ])

    await update.message.reply_text(
        "Tilni tanlang / Choose your language / Выберите язык:",
        reply_markup=kb
    )