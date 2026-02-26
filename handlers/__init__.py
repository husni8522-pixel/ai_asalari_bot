from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters

from .start_handler import start
from .text_handler import text_handler
from .callback_handler import reset_cb, lang_cb
from .admin_panel import admin_conv


def register_handlers(app):
    app.add_handler(admin_conv)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(CallbackQueryHandler(reset_cb, pattern="^reset$"))
    app.add_handler(CallbackQueryHandler(lang_cb, pattern="^lang_"))
    
