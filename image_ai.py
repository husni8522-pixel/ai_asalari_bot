import os
import json

IMAGE_DIR = "images"
KEYWORDS_FILE = "image_keywords.json"

# ===============================
# KEYWORDS LOAD
# ===============================

if os.path.exists(KEYWORDS_FILE):
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        IMAGE_KEYWORDS = json.load(f)
else:
    IMAGE_KEYWORDS = {}

# ===============================
# IMAGE SEARCH
# ===============================

def find_images_for_question(question):

    q = question.lower()

    for key, langs in IMAGE_KEYWORDS.items():

        keywords = []

        # barcha tillardagi keywordlarni birlashtirish
        if isinstance(langs, dict):
            for lang_words in langs.values():
                keywords.extend(lang_words)

        # keyword tekshirish
        for word in keywords:

            if word.lower() in q:

                # mos rasmni topish
                for file in os.listdir(IMAGE_DIR):

                    if not file.lower().endswith(("jpg","jpeg","png")):
                        continue

                    name = os.path.splitext(file)[0].lower()

                    if name.startswith(key):
                        return [os.path.join(IMAGE_DIR, file)]

    return []
