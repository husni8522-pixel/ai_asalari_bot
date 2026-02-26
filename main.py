from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler
)

from config import BOT_TOKEN
from handlers import register_handlers
from handlers.callback_handler import reset_cb, lang_cb, level_cb
from handlers.level_handler import test_answer_handler


if __name__ == "__main__":

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    register_handlers(app)

    # 🔥 TEST ENG BIRINCHI
    app.add_handler(CallbackQueryHandler(test_answer_handler, pattern="^test_"))

    # 🔥 Keyin boshqalar
    app.add_handler(CallbackQueryHandler(level_cb, pattern="^level_"))
    app.add_handler(CallbackQueryHandler(lang_cb, pattern="^lang_"))
    app.add_handler(CallbackQueryHandler(reset_cb, pattern="^reset$"))

    print("🐝 BOT ISHGA TUSHDI (MODULAR)")

    app.run_polling()