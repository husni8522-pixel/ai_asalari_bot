import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_TERMS = [
    "queen bee",
    "worker bee",
    "drone bee",
    "varroa mite",
    "nosema disease",
    "american foulbrood",
    "honey extractor",
    "bee hive",
    "bee colony",
    "bee swarm",
    "bee brood",
    "royal jelly",
    "propolis",
    "bee pollen",
    "wax foundation",
    "apiary",
    "bee smoker",
    "bee diseases",
    "wintering bees",
    "bee feeding"
]

def generate(term):

    prompt = f"""
Generate 40 professional beekeeping related synonyms and related phrases
for the term "{term}" in 3 languages.

Return ONLY pure JSON in this format:

{{
  "uz": [...],
  "ru": [...],
  "en": [...]
}}

Do not add explanations.
Do not add markdown.
Only raw JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()

    # 🔥 JSONni ajratib olish
    start = content.find("{")
    end = content.rfind("}") + 1

    clean_json = content[start:end]

    return clean_json


result = {}

for term in BASE_TERMS:
    print(f"Generating for {term}...")
    data = generate(term)

    try:
        result[term.replace(" ", "_")] = json.loads(data)
    except Exception as e:
        print("❌ JSON parse error for:", term)
        print(data)

with open("image_synonyms_big.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("🔥 10k+ synonym database generated!")