import os
import pickle
current_ad = None

# ================= BASE STORAGE (Railway Volume) =================

if os.getenv("RAILWAY_ENVIRONMENT"):
    BASE_DIR = "/app/data"
else:
    BASE_DIR = "data"

os.makedirs(BASE_DIR, exist_ok=True)

# ================= FILE PATHS =================

STATS_FILE = os.path.join(BASE_DIR, "stats.pkl")
ADS_FILE = os.path.join(BASE_DIR, "ads.pkl")
LEVELS_FILE = os.path.join(BASE_DIR, "user_levels.pkl")
INDEX_FILE = os.path.join(BASE_DIR, "index.faiss")
META_FILE = os.path.join(BASE_DIR, "meta.pkl")
IMAGE_INDEX_FILE = os.path.join(BASE_DIR, "image_index.pkl")

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

# ================= REKLAMA =================

if os.path.exists(ADS_FILE):
    try:
        ads = pickle.load(open(ADS_FILE, "rb"))
        if not isinstance(ads, list):
            ads = []
    except:
        ads = []
else:
    ads = []

