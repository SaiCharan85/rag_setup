import os

# Base configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
