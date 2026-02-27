import os
import pickle

# ================= BASE STORAGE (Railway Volume) =================

if os.getenv("RAILWAY_ENVIRONMENT"):
    BASE_DIR = "/app/data"
else:
    BASE_DIR = "data"

os.makedirs(BASE_DIR, exist_ok=True)

# ================= FILE PATHS =================

STATS_FILE = os.path.join(BASE_DIR, "stats.pkl")
LEVELS_FILE = os.path.join(BASE_DIR, "user_levels.pkl")
INDEX_FILE = os.path.join(BASE_DIR, "index.faiss")
META_FILE = os.path.join(BASE_DIR, "meta.pkl")
IMAGE_INDEX_FILE = os.path.join(BASE_DIR, "image_index.pkl")

# 🔥 REKLAMA FAYLI (persistent)
AD_FILE = os.path.join(BASE_DIR, "ad.txt")

# ================= REKLAMA =================

current_ad = None

# 🔄 Bot ishga tushganda reklama yuklanadi
if os.path.exists(AD_FILE):
    try:
        with open(AD_FILE, "r", encoding="utf-8") as f:
            current_ad = f.read().strip()
    except:
        current_ad = None

# ================= STATISTIKA =================

if os.path.exists(STATS_FILE):
    try:
        data = pickle.load(open(STATS_FILE, "rb"))
        user_stats = data.get("users", set())
        questions_log = data.get("questions", [])
    except:
        user_stats = set()
        questions_log = []
else:
    user_stats = set()
    questions_log = []

# ================= USER LEVELS (PERSISTENT) =================

if os.path.exists(LEVELS_FILE):
    try:
        user_levels = pickle.load(open(LEVELS_FILE, "rb"))
    except:
        user_levels = {}
else:
    user_levels = {}

# ================= TEMPORARY STATES =================

user_test_state = {}
user_languages = {}
user_test_cooldown = {}
chat_log = {}
user_memory = {}
admin_mode = {}
