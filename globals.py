import os
import pickle

# ================= STATISTIKA =================
STATS_FILE = "stats.pkl"

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

user_levels = {}
user_test_state = {}
user_languages = {}
user_test_cooldown = {}
# ================= CHAT LOG =================
chat_log = {}

# ================= USER MEMORY (AI CONTEXT) =================
user_memory = {}

# ================= ADMIN =================
admin_mode = {}

# ================= REKLAMA =================
ADS_FILE = "ads.pkl"

if os.path.exists(ADS_FILE):
    try:
        ads = pickle.load(open(ADS_FILE, "rb"))
        if not isinstance(ads, list):
            ads = []
    except:
        ads = []
else:
    ads = []