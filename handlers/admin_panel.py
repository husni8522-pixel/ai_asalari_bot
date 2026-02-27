import os
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


# ================= STATES =================
ADMIN_CHOOSE, UPLOAD, DELETE, ADD_AD, SEND_USER_ID, SEND_MESSAGE = range(6)


# ================= KEYBOARD =================
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


# ================= START =================
async def admin_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if u.effective_user.id != ADMIN_ID:
        await u.message.reply_text("❌ Admin emas")
        return ConversationHandler.END

    await u.message.reply_text("⚙️ Admin panel", reply_markup=admin_kb())
    return ADMIN_CHOOSE


# ================= CALLBACK =================
async def admin_cb(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()

    data = q.data

    if data == "upload":
        await q.message.reply_text("📥 Fayl yuboring")
        return UPLOAD

    if data == "delete":
        files = os.listdir(DATA_DIR)
        if not files:
            await q.message.reply_text("❌ Fayllar yo‘q")
            return ADMIN_CHOOSE

        kb = [[InlineKeyboardButton(f, callback_data=f"del::{f}")] for f in files]
        await q.message.reply_text("🗑 Fayl tanlang", reply_markup=InlineKeyboardMarkup(kb))
        return DELETE

    if data == "ad":
        await q.message.reply_text("📣 Reklama matnini kiriting:")
        return ADD_AD

    if data == "send_user":
        await q.message.reply_text("📨 User ID ni kiriting:")
        return SEND_USER_ID

    if data == "stat":
        await q.message.reply_text(
            f"👥 Userlar: {len(user_stats)}\n❓ Savollar: {len(questions_log)}"
        )
        return ADMIN_CHOOSE

    if data == "reindex":
        await q.message.reply_text("♻️ Indeks yangilanmoqda...")
        build_index()
        await q.message.reply_text("✅ Indeks yangilandi!")
        return ADMIN_CHOOSE

    if data == "exit":
        await q.message.reply_text("❌ Chiqildi")
        return ConversationHandler.END

    return ADMIN_CHOOSE


# ================= REKLAMA =================
async def admin_add_ad(u: Update, c: ContextTypes.DEFAULT_TYPE):
    text = u.message.text
    await u.message.reply_text("✅ Reklama saqlandi.")
    return ADMIN_CHOOSE


# ================= USER MESSAGE =================
async def admin_get_user_id(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = int(u.message.text)
        c.user_data["target_user_id"] = user_id
        await u.message.reply_text("✍️ Yuboriladigan xabar matnini kiriting:")
        return SEND_MESSAGE
    except ValueError:
        await u.message.reply_text("❌ Iltimos, faqat raqam kiriting.")
        return SEND_USER_ID


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


# ================= FILE =================
async def admin_file(u: Update, c: ContextTypes.DEFAULT_TYPE):
    d = u.message.document
    p = os.path.join(DATA_DIR, d.file_name)

    file = await d.get_file()
    await file.download_to_drive(p)

    await u.message.reply_text("✅ Yuklandi (Indeks yangilanmadi)")
    return ADMIN_CHOOSE


async def admin_del(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()

    f = q.data.split("::")[1]
    path = os.path.join(DATA_DIR, f)

    if os.path.exists(path):
        os.remove(path)
        await q.message.reply_text("✅ O‘chirildi (Indeks yangilanmadi)")
    else:
        await q.message.reply_text("❌ Fayl topilmadi")

    return ADMIN_CHOOSE


# ================= CANCEL =================
async def cancel_admin(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("❌ Bekor qilindi")
    return ConversationHandler.END


# ================= CONVERSATION =================
admin_conv = ConversationHandler(
    entry_points=[CommandHandler("admin", admin_start)],
    states={
        ADMIN_CHOOSE: [
            CallbackQueryHandler(
                admin_cb,
                pattern="^(upload|delete|reindex|send_user|ad|stat|exit)$"
            )
        ],
        UPLOAD: [
            MessageHandler(filters.Document.ALL, admin_file)
        ],
        DELETE: [
            CallbackQueryHandler(admin_del, pattern="^del::")
        ],
        ADD_AD: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, admin_add_ad)
        ],
        SEND_USER_ID: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, admin_get_user_id)
        ],
        SEND_MESSAGE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, admin_send_message)
        ],
    },
    fallbacks=[
        CommandHandler("cancel", cancel_admin)
    ],
)
