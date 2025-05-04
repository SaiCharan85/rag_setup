import os

# Base configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
QA_MODEL = 'google/flan-t5-base'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

# File handling configuration
DEFAULT_ENCODING = 'utf-8'
SUPPORTED_FILE_TYPES = ['.pdf', '.docx', '.pptx', '.txt']

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
