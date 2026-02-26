from asalari_words import ASALARI
from texts import TEXTS
from globals import user_languages

async def send_long_message(message, text):
    MAX_LEN = 4000  # 4096 dan biroz kichikroq xavfsiz limit
    for i in range(0, len(text), MAX_LEN):
        await message.reply_text(text[i:i+MAX_LEN])

# 🔥 Tilga mos tarjima olish
def t(uid, key, **kwargs):
    lang = user_languages.get(uid, "uz")
    text = TEXTS[key].get(lang, TEXTS[key]["uz"])
    return text.format(**kwargs)

def normalize_text(text):
    text = text.lower()

    # 🔥 Uzbek Cyrillic → Latin mapping
    mapping = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
        "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i",
        "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch",
        "ш": "sh", "щ": "sh", "ъ": "", "ь": "",
        "ы": "i", "э": "e", "ю": "yu", "я": "ya",
        "қ": "q", "ғ": "g‘", "ў": "o‘", "ҳ": "h"
    }

    for cyr, lat in mapping.items():
        text = text.replace(cyr, lat)

    return text

# 🔥 Asalarichilik so‘zlarini tekshirish
def is_asalari(text):
    text = normalize_text(text)
    return any(w in text for w in ASALARI)


# 🔥 Oddiy suhbat (salom, rahmat va hk)
def basic_chat(uid, text):

    txt = text.lower()
    words = txt.split()  # 🔥 substring emas, so‘zlar bo‘yicha tekshiruv
    lang = user_languages.get(uid, "uz")

    responses = {

        "greet": {
            "uz": "Assalomu alaykum 😊 Marhamat, savolingizni bering.",
            "ru": "Здравствуйте 😊 Задайте ваш вопрос.",
            "en": "Hello 😊 Please ask your question."
        },

        "bye": {
            "uz": "Xayr! Sizni yana kutib qolamiz 😊",
            "ru": "До свидания! Будем рады видеть вас снова 😊",
            "en": "Goodbye! We hope to see you again 😊"
        },

        "thanks": {
            "uz": "Arzimaydi 😊 Yordam bera olganimdan xursandman.",
            "ru": "Не за что 😊 Рад был помочь.",
            "en": "You're welcome 😊 Happy to help."
        },

        "creator": {
            "uz": "Men Husniddin Zaripov tomonidan yaratilgan botman 😊",
            "ru": "Я создан Хусниддином Зариповым 😊",
            "en": "I was created by Husniddin Zaripov 😊"
        },

        "contact": {
            "uz": "Aloqa uchun: +998973850026 📞",
            "ru": "Связь: +998973850026 📞",
            "en": "Contact: +998973850026 📞"
        }
    }

    # 🔥 SO‘Z BO‘YICHA TEKSHIRUV
    greetings = ["salom", "assalomu", "hello", "hi", "привет"]
    if any(word in greetings for word in words):
        return responses["greet"][lang]

    bye_words = ["xayr", "hayr", "goodbye", "bye", "пока"]
    if any(word in bye_words for word in words):
        return responses["bye"][lang]

    thanks_words = ["rahmat", "raxmat", "спасибо", "thanks"]
    if any(word in thanks_words for word in words):
        return responses["thanks"][lang]

    creator_words = ["kim", "yaratgan", "tuzgan", "who"]
    if "kim" in words and ("yaratgan" in words or "tuzgan" in words):
        return responses["creator"][lang]

    if any(word in ["aloqa", "contact"] for word in words):
        return responses["contact"][lang]

    return None