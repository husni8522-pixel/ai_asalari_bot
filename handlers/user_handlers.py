from telegram import Update
from telegram.ext import ContextTypes
from ai import ai_answer

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    txt = update.message.text

    answer = ai_answer(uid, txt)

    await update.message.reply_text(answer)
