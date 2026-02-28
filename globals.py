import os
import pickle

# ================= FILE PATHS =================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

STATS_FILE = os.path.join(DATA_DIR, "stats.pkl")
ADS_FILE = os.path.join(DATA_DIR, "ads.pkl")
LEVELS_FILE = os.path.join(DATA_DIR, "user_levels.pkl")

# ================= ADS LIST (PERSISTENT) =================

if os.path.exists(ADS_FILE):
    try:
        ads = pickle.load(open(ADS_FILE, "rb"))
        if not isinstance(ads, list):
            ads = []
    except:
        ads = []
else:
    ads = []

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

# ================= USER LEVELS =================

if os.path.exists(LEVELS_FILE):
    try:
        user_levels = pickle.load(open(LEVELS_FILE, "rb"))
    except:
        user_levels = {}
else:
    user_levels = {}

# ================= TEMP =================

user_test_state = {}
user_languages = {}
user_test_cooldown = {}
chat_log = {}
user_memory = {}
