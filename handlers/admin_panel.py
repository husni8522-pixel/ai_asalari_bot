import os
import globals

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters
)

from config import DATA_DIR, ADMIN_ID
from indexer import build_index
from globals import user_stats, questions_log


# 🔥 ADD_AD qo‘shildi
ADMIN_CHOOSE, UPLOAD, DELETE, SEND_USER_ID, SEND_MESSAGE, ADD_AD = range(6)


def admin_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📥 Fayl yuklash", callback_data="upload")],
        [InlineKeyboardButton("🗑 Fayl o‘chirish", callback_data="delete")],
        [InlineKeyboardButton("♻️ Reindex", callback_data="reindex")],
        [InlineKeyboardButton("📨 Userga xabar", callback_data="send_user")],
        [InlineKeyboardButton("📣 Reklama", callback_data="ad")],
        [InlineKeyboardButton("📊 Statistika", callback_data="stat")],
        [InlineKeyboardButton("❌ Chiqish", callback_data="exit")]
    ])


async def admin_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if u.effective_user.id != ADMIN_ID:
        await u.message.reply_text("❌ Admin emas")
        return ConversationHandler.END

    await u.message.reply_text("⚙️ Admin panel", reply_markup=admin_kb())
    return ADMIN_CHOOSE


async def admin_cb(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()

    if q.data == "upload":
        await q.message.reply_text("📥 Fayl yuboring")
        return UPLOAD

    if q.data == "delete":
        files = os.listdir(DATA_DIR)
        kb = [[InlineKeyboardButton(f, callback_data=f"del::{f}")] for f in files]
        await q.message.reply_text("🗑 Fayl tanlang", reply_markup=InlineKeyboardMarkup(kb))
        return DELETE

    if q.data == "send_user":
        await q.message.reply_text("📨 User ID ni kiriting:")
        return SEND_USER_ID

    # 🔥 REKLAMA QO‘SHISH
    if q.data == "ad":
        await q.message.reply_text("📣 Reklama matnini kiriting:")
        return ADD_AD

    if q.data == "stat":
        await q.message.reply_text(
            f"👥 Userlar: {len(user_stats)}\n❓ Savollar: {len(questions_log)}"
        )
        return ADMIN_CHOOSE

    if q.data == "reindex":
        await q.message.reply_text("♻️ Indeks yangilanmoqda...")
        build_index()
        await q.message.reply_text("✅ Indeks yangilandi!")
        return ADMIN_CHOOSE

    if q.data == "exit":
        await q.message.reply_text("❌ Chiqildi")
        return ConversationHandler.END

    return ADMIN_CHOOSE


# 🔥 REKLAMA SAQLASH (PERSISTENT)
async def admin_add_ad(u: Update, c: ContextTypes.DEFAULT_TYPE):
    text = u.message.text.strip()

    # RAM ga saqlash
    globals.current_ad = text

    # Faylga saqlash
    with open(globals.AD_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    await u.message.reply_text("✅ Reklama saqlandi va doimiy qilindi.")
    return ADMIN_CHOOSE


# 📌 1-qadam: User ID olish
async def admin_get_user_id(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = int(u.message.text)
        c.user_data["target_user_id"] = user_id
        await u.message.reply_text("✍️ Yuboriladigan xabar matnini kiriting:")
        return SEND_MESSAGE
    except ValueError:
        await u.message.reply_text("❌ Iltimos, faqat raqam kiriting.")
        return SEND_USER_ID


# 📌 2-qadam: Xabar yuborish
async def admin_send_message(u: Update, c: ContextTypes.DEFAULT_TYPE):
    target_id = c.user_data.get("target_user_id")

    if not target_id:
        await u.message.reply_text("❌ User ID topilmadi.")
        return ConversationHandler.END

    text = u.message.text

    try:
        await c.bot.send_message(chat_id=target_id, text=f"📩 Admin xabari:\n\n{text}")
        await u.message.reply_text("✅ Xabar yuborildi.")
    except Exception as e:
        await u.message.reply_text(f"❌ Yuborib bo‘lmadi:\n{e}")

    c.user_data.clear()
    return ADMIN_CHOOSE


async def admin_file(u: Update, c: ContextTypes.DEFAULT_TYPE):
    d = u.message.document
    p = os.path.join(DATA_DIR, d.file_name)
    await (await d.get_file()).download_to_drive(p)
    build_index()
    await u.message.reply_text("✅ Yuklandi va indeks yangilandi")
    return ADMIN_CHOOSE


async def admin_del(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    f = q.data.split("::")[1]
    os.remove(os.path.join(DATA_DIR, f))
    build_index()
    await q.message.reply_text("✅ O‘chirildi")
    return ADMIN_CHOOSE


admin_conv = ConversationHandler(
    entry_points=[CommandHandler("admin", admin_start)],
    states={
        ADMIN_CHOOSE: [CallbackQueryHandler(admin_cb)],
        UPLOAD: [MessageHandler(filters.Document.ALL, admin_file)],
        DELETE: [CallbackQueryHandler(admin_del, pattern="^del::")],
        SEND_USER_ID: [MessageHandler(filters.TEXT & ~filters.COMMAND, admin_get_user_id)],
        SEND_MESSAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, admin_send_message)],
        ADD_AD: [MessageHandler(filters.TEXT & ~filters.COMMAND, admin_add_ad)],
    },
    fallbacks=[]
)


