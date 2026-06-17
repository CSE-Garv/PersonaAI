import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

DB_PATH = "./chroma_db"
DATA_PATH = "./data"
JSON_PATH = "./personalities.json"

# Semantic Cache
CACHE_SIMILARITY_THRESHOLD = 0.92  # Cosine similarity threshold for cache hits

# Simple Admin Credentials (Hardcoded for demo)
ADMIN_USER = "admin"
ADMIN_PASS = "lumos"