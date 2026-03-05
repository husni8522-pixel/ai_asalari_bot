import os
import json
from openai import OpenAI

IMAGE_DIR="images"
TAG_FILE="image_tags.json"

client=OpenAI()

# =========================
# TAG FILE LOAD
# =========================

if os.path.exists(TAG_FILE):
    with open(TAG_FILE,"r",encoding="utf-8") as f:
        IMAGE_TAGS=json.load(f)
else:
    IMAGE_TAGS={}

# =========================
# AUTO TAG GENERATION
# =========================

def generate_tags(image_name):

    prompt=f"""
Generate search keywords for this beekeeping image.

Image name: {image_name}

Rules:
- include english
- include russian
- include uzbek
- include scientific terms
- 10 keywords
Return list only.
"""

    r=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":"You generate search tags for beekeeping images."},
            {"role":"user","content":prompt}
        ],
        temperature=0
    )

    text=r.choices[0].message.content

    tags=[t.strip() for t in text.split("\n") if t.strip()]

    return tags


# =========================
# BUILD TAG DATABASE
# =========================

def build_image_tags():

    global IMAGE_TAGS

    for file in os.listdir(IMAGE_DIR):

        if not file.endswith(("jpg","png","jpeg")):
            continue

        if file in IMAGE_TAGS:
            continue

        tags=generate_tags(file)

        IMAGE_TAGS[file]=tags

        print("Tagged:",file)

    with open(TAG_FILE,"w",encoding="utf-8") as f:
        json.dump(IMAGE_TAGS,f,ensure_ascii=False,indent=2)


# =========================
# IMAGE SEARCH
# =========================

def find_images_for_question(question):

    q=question.lower()

    results=[]

    for img,tags in IMAGE_TAGS.items():

        for tag in tags:

            if tag.lower() in q:
                results.append(os.path.join(IMAGE_DIR,img))

    return results[:3]
