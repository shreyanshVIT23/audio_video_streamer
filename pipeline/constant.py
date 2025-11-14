from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv("../.env")


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
AUDIO_DB_PATH = Path("./known_voices/")
AUDIO_THRESHOLD = 0.45
SAMPLE_RATE = 16000
VIDEO_DB_PATH = Path("./known_faces/")
VIDEO_THRESHOLD = 0.4
