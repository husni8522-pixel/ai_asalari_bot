import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = "data"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"
ADS_FILE = "ads.pkl"

CHUNK_SIZE = 1000
BATCH_SIZE = 32
TOP_K = 8

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))

client = OpenAI(api_key=OPENAI_KEY)
