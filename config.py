import os
from openai import OpenAI

# ================= PATH =================
DATA_DIR = "data"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"
ADS_FILE = "ads.pkl"

CHUNK_SIZE = 1000
BATCH_SIZE = 32
TOP_K = 8

# ================= ENV =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

# ================= OPENAI CLIENT =================
client = OpenAI(api_key=OPENAI_KEY)

# ================= DEBUG (optional) =================
print("ADMIN_ID:", ADMIN_ID)
